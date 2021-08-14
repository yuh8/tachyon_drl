import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .CONSTS import (DROPOUT_RATE, EMBEDDING_SIZE, FFD_SIZE, MAX_MOL_LEN,
                     MOL_DICT, NUM_HEADS)


def batch_matmul(batch_mat, mat):
    batch_mat_shape = tf.shape(batch_mat)
    mul_mat = tf.reshape(tf.reshape(batch_mat, [-1, batch_mat_shape[-1]]) @ mat,
                         batch_mat_shape)
    return mul_mat


def get_padding_mask(x):
    # [BATCH, MAX_MOL_LEN]
    valid_bool = tf.less(x, len(MOL_DICT))
    # [BATCH, MAX_MOL_LEN]
    padding_mask = tf.where(valid_bool, 1, 0)
    # [BATCH, 1, MAX_MOL_LEN]
    padding_mask = tf.expand_dims(padding_mask, axis=1)
    # [BATCH, 1, 1, MAX_MOL_LEN]
    padding_mask = tf.expand_dims(padding_mask, axis=1)
    return padding_mask


def get_position_embedding():
    # [0,1,...,MAX_MOL_LEN]
    position_range = tf.range(0, MAX_MOL_LEN)
    # [MAX_MOL_LEN, EMBEDDING_SIZE]
    position_embedding = layers.Embedding(MAX_MOL_LEN, EMBEDDING_SIZE)(position_range)
    return position_embedding


def get_token_embedding(encoded_token_inputs):
    '''
    encoded_token_input: [BATCH, MAX_MOL_LEN]
    '''
    one_hot_depth = len(MOL_DICT) + 1
    # [BATCH, MAX_MOL_LEN, DICT_LEN]
    token_embedding = tf.one_hot(encoded_token_inputs, one_hot_depth,
                                 on_value=1.0, off_value=0.0, axis=-1)
    # [BATCH, MAX_MOL_LEN, EMBEDDING_SIZE]
    token_embedding = layers.Dense(EMBEDDING_SIZE, activation="relu")(token_embedding)
    position_embedding = get_position_embedding()
    token_embedding = token_embedding + position_embedding
    return token_embedding


def get_singlehead_attention(Q, K, V, mask):
    # [BATCH_SIZE, NUM_HEADS, DEPTH, MAX_MOL_LEN]
    K_transpose = tf.transpose(K, perm=[0, 1, 3, 2])
    # [BATCH, NUM_HEADS, MAX_MOL_LEN, MAX_MOL_LEN]
    scaled_attention = tf.math.divide(Q @ K_transpose,
                                      tf.math.sqrt(float(EMBEDDING_SIZE)))
    invalid_mask = tf.where(tf.equal(mask, 0), -1e10, 1)
    scaled_attention = scaled_attention * invalid_mask
    # [BATCH_SIZE, NUM_HEADS, MAX_MOL_LEN, DEPTH]
    scaled_attention = tf.nn.softmax(scaled_attention, axis=-1) @ V
    return scaled_attention


class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        assert EMBEDDING_SIZE % num_heads == 0
        self.depth = EMBEDDING_SIZE // num_heads

        # create key, query, value matrix
        initializer = keras.initializers.GlorotUniform()
        self.W_q = tf.Variable(initializer([EMBEDDING_SIZE, EMBEDDING_SIZE]))
        self.W_k = tf.Variable(initializer([EMBEDDING_SIZE, EMBEDDING_SIZE]))
        self.W_v = tf.Variable(initializer([EMBEDDING_SIZE, EMBEDDING_SIZE]))
        self.dense = layers.Dense(EMBEDDING_SIZE)
        self.batch_norm = layers.BatchNormalization()

    def split_heads(self, x):
        x = tf.reshape(x, (-1, MAX_MOL_LEN, self.num_heads, self.depth))
        # [BATCH, NUM_HEADS, MAX_MOL_LEN, DEPTH]
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, token_embedding, mask):
        # [BATCH, MAX_MOL_LEN, EMBEDDING_SIZE]
        Q = batch_matmul(token_embedding, self.W_q)
        K = batch_matmul(token_embedding, self.W_k)
        V = batch_matmul(token_embedding, self.W_v)

        # [BATCH, NUM_HEADS, MAX_MOL_LEN, DEPTH]
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # [BATCH_SIZE, NUM_HEADS, MAX_MOL_LEN, DEPTH]
        scaled_attention = get_singlehead_attention(Q, K, V, mask)
        # [BATCH_SIZE, MAX_MOL_LEN, NUM_HEADS, DEPTH]
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # [BATCH_SIZE, MAX_MOL_LEN, EMBEDDING_SIZE]
        concat_attention = tf.reshape(scaled_attention, (-1, MAX_MOL_LEN, EMBEDDING_SIZE))
        concat_attention = self.dense(concat_attention)
        concat_attention = self.batch_norm(concat_attention)
        return concat_attention


def get_point_wise_feed_forward_network(dff):
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        layers.Dense(EMBEDDING_SIZE)  # (batch_size, seq_len, d_model)
    ])


class BERTblock(layers.Layer):
    def __init__(self):
        super(BERTblock, self).__init__()
        # self.mha = MultiHeadAttention(NUM_HEADS)
        self.mha = layers.MultiHeadAttention(NUM_HEADS, EMBEDDING_SIZE)
        self.ffn = get_point_wise_feed_forward_network(FFD_SIZE)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.3)
        self.dropout2 = layers.Dropout(0.3)

    def call(self, x, padding_mask):
        '''
        x: [BATCH_SIZE, MAX_MOL_LEN, EMBEDDING_SIZE]
        '''
        # [BATCH_SIZE, MAX_MOL_LEN, EMBEDDING_SIZE]
        attn1 = self.mha(x, x, attention_mask=padding_mask)
        attn1 = self.dropout1(attn1)
        # residual connection
        out1 = attn1 + x
        out1 = self.layernorm1(out1)

        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        # residual connection
        ffn_out = ffn_out + out1
        ffn_out = self.layernorm2(ffn_out)
        return ffn_out
