import numpy as np
from src.CONSTS import MAX_MOL_LEN, MOL_DICT


def sample_single_token_from_probs(probs):
    indices = np.arange(len(MOL_DICT) + 1)
    return np.random.choice(indices, p=probs)


def generate_smile(gen_net):
    start_tokens = ["G"]
    start_tokens_ids = [MOL_DICT.index("G")]
    tokens_generated = []
    tokens_idx_generated = []
    while len(tokens_generated) < MAX_MOL_LEN:
        pad_len = MAX_MOL_LEN - len(start_tokens)
        sample_index = len(start_tokens) - 1
        encoded_token = [MOL_DICT.index(token) for token in start_tokens]
        if pad_len <= 0:
            x = encoded_token[:MAX_MOL_LEN]
            sample_index = MAX_MOL_LEN - 1
        else:
            x = encoded_token + [len(MOL_DICT)] * pad_len
        x = np.array([x])
        # [1, MAX_MOL_LEN, DICT_LEN + 1]
        y = gen_net.predict(x)
        sample_token_idx = sample_single_token_from_probs(y[0][sample_index])
        # skip padding generation
        if sample_token_idx == len(MOL_DICT):
            continue
        sample_token = MOL_DICT[sample_token_idx]
        # skip start token generation
        if sample_token == "G":
            continue
        tokens_generated.append(sample_token)
        tokens_idx_generated.append(sample_token_idx)
        if (tokens_generated[-1] == "E") and (len(tokens_generated) > 2):
            break
        start_tokens.append(sample_token)
        start_tokens_ids.append(sample_token_idx)

    if len(tokens_idx_generated) < MAX_MOL_LEN:
        pad_len = MAX_MOL_LEN - len(tokens_idx_generated)
        tokens_idx_generated = tokens_idx_generated + [len(MOL_DICT)] * pad_len
    else:
        tokens_idx_generated = tokens_idx_generated[:MAX_MOL_LEN]

    if len(start_tokens_ids) < MAX_MOL_LEN:
        pad_len = MAX_MOL_LEN - len(start_tokens_ids)
        start_tokens_ids = start_tokens_ids + [len(MOL_DICT)] * pad_len
    else:
        start_tokens_ids = start_tokens_ids[:MAX_MOL_LEN]

    return start_tokens, start_tokens_ids, tokens_generated, tokens_idx_generated
