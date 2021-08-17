import numpy as np
from scipy.special import softmax

from embed_utils import get_encoded_smi
from .CONSTS import MAX_MOL_LEN, MOL_DICT


def sample_single_token_from_logits(logits):
    indices = np.arange(len(MOL_DICT) + 1)
    probs = softmax(logits)
    return np.random.choice(indices, p=probs)


def generate_smile(caspar_net):
    start_tokens = ["G"]
    tokens_generated = []
    tokens_idx_generated = []
    while (len(tokens_generated) < MAX_MOL_LEN) and (tokens_generated[-1] != "E"):
        pad_len = MAX_MOL_LEN - len(start_tokens)
        sample_index = len(start_tokens) - 1
        encoded_token = [MOL_DICT.index(token) for token in start_tokens]
        if pad_len <= 0:
            x = encoded_token[:MAX_MOL_LEN]
            sample_index = MAX_MOL_LEN - 1
        else:
            x = encoded_token + [len(MAX_MOL_LEN)] * pad_len
        x = np.array([x])
        # [1, MAX_MOL_LEN, DICT_LEN + 1]
        y = caspar_net.predict(x)
        sample_token_idx = sample_single_token_from_logits(y[0][sample_index])
        # skip padding generation
        if sample_token_idx == len(MOL_DICT):
            continue
        sample_token = MOL_DICT[sample_token_idx]
        # skip start token generation
        if sample_token == "G":
            continue
        tokens_generated.append(sample_token)
        start_tokens.append(sample_token)
        tokens_idx_generated.append(sample_token_idx)

    if tokens_generated[-1] == "E":
        tokens_generated = tokens_generated[:-1]
        tokens_idx_generated = tokens_idx_generated[:-1]

    if len(tokens_idx_generated) < MAX_MOL_LEN:
        pad_len = MAX_MOL_LEN - len(tokens_idx_generated)
        tokens_idx_generated = tokens_idx_generated + [len(MAX_MOL_LEN)] * pad_len

    return tokens_generated, tokens_idx_generated
