import tensorflow as tf
from tensorflow.keras import layers
from .CONSTS import (EMBEDDING_SIZE, FFD_SIZE, MAX_MOL_LEN,
                     MOL_DICT, NUM_HEADS, DROPOUT_RATE)


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
    return padding_mask


def get_causal_attention_mask():
    # [MAX_MOL_LEN, MAX_MOL_LEN]
    causal_mask = tf.linalg.band_part(tf.ones((MAX_MOL_LEN, MAX_MOL_LEN)), -1, 0)
    causal_mask = tf.cast(causal_mask, tf.int32)
    # [1, MAX_MOL_LEN, MAX_MOL_LEN]
    causal_mask = tf.expand_dims(causal_mask, axis=0)
    return causal_mask


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


def get_gpt_block(x, padding_mask, causal_mask):
    mask = tf.minimum(causal_mask, padding_mask)
    mha = layers.MultiHeadAttention(NUM_HEADS, EMBEDDING_SIZE)
    # [BATCH_SIZE, MAX_MOL_LEN, EMBEDDING_SIZE]
    attn = mha(x, x, x, attention_mask=mask)
    attn = layers.Dropout(DROPOUT_RATE)(attn)
    attn = attn + x
    attn = layers.LayerNormalization(epsilon=1e-6)(attn)
    fc_out = get_point_wise_feed_forward_network(FFD_SIZE)(attn)
    fc_out = layers.Dropout(DROPOUT_RATE)(fc_out)
    fc_out = fc_out + attn
    fc_out = layers.LayerNormalization(epsilon=1e-6)(fc_out)
    return fc_out
