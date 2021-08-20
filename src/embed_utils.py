import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from .CONSTS import (EMBEDDING_SIZE_GEN, EMBEDDING_SIZE_PRED,
                     FFD_SIZE_GEN, FFD_SIZE_PRED,
                     NUM_HEADS_GEN, NUM_HEADS_PRED,
                     NUM_LAYERS_GEN, NUM_LAYERS_PRED,
                     MOL_DICT, MAX_MOL_LEN, DROPOUT_RATE)


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


def get_position_embedding(embed_size):
    # [0,1,...,MAX_MOL_LEN]
    position_range = tf.range(0, MAX_MOL_LEN)
    # [MAX_MOL_LEN, EMBEDDING_SIZE]
    position_embedding = layers.Embedding(MAX_MOL_LEN, embed_size)(position_range)
    return position_embedding


def get_token_embedding(encoded_token_inputs, embed_size):
    '''
    encoded_token_input: [BATCH, MAX_MOL_LEN]
    '''
    # [BATCH, MAX_MOL_LEN, EMBEDDING_SIZE]
    token_embedding = layers.Embedding(len(MOL_DICT) + 1, embed_size)(encoded_token_inputs)
    position_embedding = get_position_embedding(embed_size)
    token_embedding = token_embedding + position_embedding
    return token_embedding


def get_point_wise_feed_forward_network(dff, embed_size):
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        layers.Dense(embed_size)  # (batch_size, seq_len, d_model)
    ])


def get_attention_block(x, mask, num_heads, embed_size, ffd_size):
    mha = layers.MultiHeadAttention(num_heads, embed_size)
    # [BATCH_SIZE, MAX_MOL_LEN, EMBEDDING_SIZE]
    attn = mha(x, x, x, attention_mask=mask)
    attn = layers.Dropout(DROPOUT_RATE)(attn)
    attn = attn + x
    attn = layers.LayerNormalization(epsilon=1e-6)(attn)
    fc_out = get_point_wise_feed_forward_network(ffd_size, embed_size)(attn)
    fc_out = layers.Dropout(DROPOUT_RATE)(fc_out)
    fc_out = fc_out + attn
    fc_out = layers.LayerNormalization(epsilon=1e-6)(fc_out)
    return fc_out


def get_gpt_block(x, padding_mask, causal_mask):
    mask = tf.minimum(causal_mask, padding_mask)
    fc_out = get_attention_block(x, mask,
                                 NUM_HEADS_GEN,
                                 EMBEDDING_SIZE_GEN,
                                 FFD_SIZE_GEN)
    return fc_out


def get_decoder(x, padding_mask, causal_mask):
    for _ in range(NUM_LAYERS_GEN):
        x = get_gpt_block(x, padding_mask, causal_mask)
    return x


def get_bert_block(x, mask):
    fc_out = get_attention_block(x, mask,
                                 NUM_HEADS_PRED,
                                 EMBEDDING_SIZE_PRED,
                                 FFD_SIZE_PRED)
    return fc_out


def get_encoder(x, padding_mask):
    for _ in range(NUM_LAYERS_PRED):
        x = get_bert_block(x, padding_mask)
    return x


def get_generator_model():
    # [BATCH, MAX_MOL_LEN]
    smi_inputs = layers.Input(shape=(MAX_MOL_LEN,), dtype=np.int32)
    token_embedding = get_token_embedding(smi_inputs, EMBEDDING_SIZE_GEN)
    padding_mask = get_padding_mask(smi_inputs)
    causal_mask = get_causal_attention_mask()
    gen_out = get_decoder(token_embedding, padding_mask, causal_mask)
    # [BATCH, MAX_MOL_LEN, DICT_LEN]
    logits = layers.Dense(len(MOL_DICT) + 1)(gen_out)
    return smi_inputs, logits


def get_predictor_model():
    # [BATCH, MAX_MOL_LEN]
    smi_inputs = layers.Input(shape=(MAX_MOL_LEN,), dtype=np.int32)
    token_embedding = get_token_embedding(smi_inputs, EMBEDDING_SIZE_PRED)
    padding_mask = get_padding_mask(smi_inputs)
    #[BATCH, MAX_MOL_LEN, EMBEDDING_SIZE]
    y_pred = get_encoder(token_embedding, padding_mask)
    #[BATCH, EMBEDDING_SIZE]
    y_pred = tf.reshape(y_pred, (-1, MAX_MOL_LEN * EMBEDDING_SIZE_PRED))
    y_pred = layers.Dense(EMBEDDING_SIZE_PRED, activation='relu')(y_pred)
    # [BATCH, 1]
    y_pred = layers.Dense(1, activation=None)(y_pred)
    return smi_inputs, y_pred
