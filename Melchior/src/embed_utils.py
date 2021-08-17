import tensorflow as tf
from tensorflow.keras import layers
from .CONSTS import (DROPOUT_RATE, EMBEDDING_SIZE, FFD_SIZE, MAX_MOL_LEN,
                     MOL_DICT, NUM_HEADS)


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


def get_bert_block(x, padding_mask):
    mha = layers.MultiHeadAttention(NUM_HEADS, EMBEDDING_SIZE)
    attn = mha(x, x, x, attention_mask=padding_mask)
    attn = layers.Dropout(DROPOUT_RATE)(attn)
    attn = attn + x
    attn = layers.LayerNormalization(epsilon=1e-6)(attn)
    fc_out = layers.Dense(FFD_SIZE, activation='relu')(attn)
    fc_out = layers.Dense(EMBEDDING_SIZE, activation='relu')(fc_out)
    fc_out = layers.Dropout(DROPOUT_RATE)(fc_out)
    fc_out = fc_out + attn
    fc_out = layers.LayerNormalization(epsilon=1e-6)(fc_out)
    return fc_out
