import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.CONSTS import MOL_DICT


EMBEDDING_SIZE = 128
FFD_SIZE = 128
MAX_MOL_LEN = 65
NUM_HEADS = 1
NUM_LAYERS = 2
DROPOUT_RATE = 0.1


def get_padding_mask(x):
    # [BATCH, MAX_MOL_LEN]
    valid_bool = tf.less(x, len(MOL_DICT))
    # [BATCH, MAX_MOL_LEN]
    padding_mask = tf.where(valid_bool, 1, 0)
    # [BATCH, 1, MAX_MOL_LEN]
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
    # [BATCH, MAX_MOL_LEN, EMBEDDING_SIZE]
    token_embedding = layers.Embedding(len(MOL_DICT) + 1, EMBEDDING_SIZE)(encoded_token_inputs)
    position_embedding = get_position_embedding()
    token_embedding = token_embedding + position_embedding
    return token_embedding


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
        self.dropout1 = layers.Dropout(DROPOUT_RATE)
        self.dropout2 = layers.Dropout(DROPOUT_RATE)

    def call(self, x, padding_mask):
        '''
        x: [BATCH_SIZE, MAX_MOL_LEN, EMBEDDING_SIZE]
        '''
        # [BATCH_SIZE, MAX_MOL_LEN, EMBEDDING_SIZE]
        attn1 = self.mha(x, x, x, attention_mask=padding_mask)
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


class MelchiorLayer(layers.Layer):
    def __init__(self, num_layers):
        super(MelchiorLayer, self).__init__()
        self.num_layers = num_layers
        self.bert_layers = [BERTblock() for _ in range(num_layers)]

    def call(self, x, padding_mask):
        for i in range(self.num_layers):
            x = self.bert_layers[i](x, padding_mask)

        #[BATCH, MAX_MOL_LEN, EMBEDDING_SIZE]
        return x


def get_melchior_model():
    # [BATCH, MAX_MOL_LEN]
    smi_inputs = layers.Input(shape=(MAX_MOL_LEN,), dtype=np.int32)
    token_embedding = get_token_embedding(smi_inputs)
    padding_mask = get_padding_mask(smi_inputs)
    #[BATCH, MAX_MOL_LEN, EMBEDDING_SIZE]
    melchior_out = MelchiorLayer(NUM_LAYERS)(token_embedding, padding_mask)
    #[BATCH, EMBEDDING_SIZE]
    # melchior_out = tf.reduce_max(melchior_out, axis=1)
    melchior_out = tf.reshape(melchior_out, (-1, MAX_MOL_LEN * EMBEDDING_SIZE))
    y_pred = layers.Dense(EMBEDDING_SIZE, activation='relu')(melchior_out)
    # [BATCH, 1]
    y_pred = layers.Dense(1, activation=None)(melchior_out)
    model = keras.Model(smi_inputs, y_pred)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError())
    model.load_weights("melchior_model/")
    return model
