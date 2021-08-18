import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from .embed_utils import get_encoded_smi
from .CONSTS import MAX_MOL_LEN, MOL_DICT
RDLogger.DisableLog('rdApp.*')


def get_diversity(smi_batch_a, smi_batch_b):
    td = 0
    fps_A = []
    for smi in smi_batch_a:
        try:
            mol = Chem.MolFromSmiles(smi)
            fps_A.append(Chem.RDKFingerprint(mol))
        except:
            print('ERROR: Invalid SMILES!')

    fps_B = []
    for smi in smi_batch_b:
        try:
            mol = Chem.MolFromSmiles(smi)
            fps_B.append(Chem.RDKFingerprint(mol))
        except:
            print('ERROR: Invalid SMILES!')

    for a in fps_A:
        for b in fps_B:
            ts = 1 - DataStructs.FingerprintSimilarity(a, b)
            td += ts

    td = td / (len(fps_A) * len(fps_B))
    print("RDK distance: " + str(td))
    return td


def is_valid_smile(smi):
    if Chem.MolFromSmiles(smi) is None:
        return False
    return True


def get_terminal_reward(smi_token_list, smi_bank, melchior_net):
    if smi_token_list[-1] == "E":
        token_list = smi_token_list[:-1]
    else:
        token_list = smi_token_list

    encoded_smi = get_encoded_smi(token_list)

    if len(smi_token_list) == MAX_MOL_LEN:
        T = MAX_MOL_LEN - 1
    else:
        # include the "E" at the end
        T = np.where(encoded_smi[0] < len(MOL_DICT))[0][-1] + 2

    smi = "".join(token_list)
    if not is_valid_smile(smi):
        return 0, T

    scaled_reward = melchior_net.predict(encoded_smi)[0][0]
    diversity = 1

    if len(smi_bank) >= 20:
        diversity = get_diversity([smi], smi_bank)

    if diversity < 0.75:
        rew_div = 0.9
    elif diversity > 0.9:
        rew_div = 1
    else:
        rew_div = 1

    return scaled_reward * rew_div, T


def get_time_distributed_rewards(reward, gamma, T):
    # Most reward assigned to the last time step
    t_distribute = T - np.arange(T)
    return (gamma**t_distribute) * reward


def get_padded_reward_vec(reward, gamma, T):
    distributed_reward = get_time_distributed_rewards(reward, gamma, T)
    if len(distributed_reward) < MAX_MOL_LEN:
        pad_len = MAX_MOL_LEN - len(distributed_reward)
        padded_reward = np.pad(distributed_reward,
                               (0, pad_len),
                               'constant', constant_values=0)
        return padded_reward

    return distributed_reward[:MAX_MOL_LEN]
