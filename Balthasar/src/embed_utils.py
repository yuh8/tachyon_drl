import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from .CONSTS import MOL_DICT, MAX_MOL_LEN


def import_tf_model(model_dir, custom_func=None):
    try:
        if not custom_func:
            net = models.load_model(model_dir)
        else:
            net = models.load_model(model_dir, custom_objects={'loss_func':
                                                               custom_func})
    except Exception:
        return None
    return net


def np_one_hot(array):
    n_values = np.max(array) + 1
    return np.eye(n_values)[array]


def get_encoded_smi(smi_token_list):
    encoded_smi = []
    for char in smi_token_list:
        encoded_smi.append(MOL_DICT.index(char))

    if len(encoded_smi) <= MAX_MOL_LEN:
        num_pads = MAX_MOL_LEN - len(encoded_smi)
        # 39 is the padding number which will be masked
        encoded_smi += [len(MOL_DICT)] * num_pads
    else:
        encoded_smi = encoded_smi[:MAX_MOL_LEN]
    encoded_smi = np.array(encoded_smi)
    return encoded_smi[np.newaxis, :]


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
