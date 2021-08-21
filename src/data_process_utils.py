import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from .CONSTS import MAX_MOL_LEN, MOL_DICT
RDLogger.DisableLog('rdApp.*')


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def standardize_smi(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        standardized_smi = Chem.MolToSmiles(mol)
    except:
        return np.nan

    return standardized_smi


def parse_mol_to_smi(mol):
    try:
        standardized_smi = Chem.MolToSmiles(mol)
    except:
        return None

    return standardized_smi


def get_df_from_smi(smi_path):
    x = []
    y = []
    data = []
    with Chem.SmilesMolSupplier(smi_path, nameColumn=-1, titleLine=False) as suppl:
        for mol in suppl:
            if not parse_mol_to_smi(mol):
                continue
            # G and E are speical tokens are start and end, respectively
            standardized_smi = MOL_DICT[-2] + parse_mol_to_smi(mol) + MOL_DICT[-1]
            x.append(standardized_smi[:-1])
            y.append(standardized_smi[1:])
            data.append(standardized_smi)

    df = pd.DataFrame()
    df.loc[:, 'X'] = x
    df.loc[:, 'Y'] = y
    df.loc[:, 'Data'] = data
    return df


def tokenize_smi(smi):
    N = len(smi)
    i = 0
    token = []
    while i < N:
        for symbol in MOL_DICT:
            if symbol == smi[i:i + len(symbol)]:
                token.append(symbol)
                i += len(symbol)
                break
    return token


def get_encoded_smi(smi):
    tokenized_smi = tokenize_smi(smi)
    encoded_smi = []
    for char in tokenized_smi:
        encoded_smi.append(MOL_DICT.index(char))

    if len(encoded_smi) <= MAX_MOL_LEN:
        num_pads = MAX_MOL_LEN - len(encoded_smi)
        # len(MOL_DICT) is the padding number which will be masked
        encoded_smi += [len(MOL_DICT)] * num_pads
    else:
        encoded_smi = encoded_smi[:MAX_MOL_LEN]
    return encoded_smi


def get_encoded_smi_rl(smi_token_list):
    encoded_smi = []
    for char in smi_token_list:
        encoded_smi.append(MOL_DICT.index(char))

    if len(encoded_smi) <= MAX_MOL_LEN:
        num_pads = MAX_MOL_LEN - len(encoded_smi)
        # len(MOL_DICT) is the padding number which will be masked
        encoded_smi += [len(MOL_DICT)] * num_pads
    else:
        encoded_smi = encoded_smi[:MAX_MOL_LEN]
    encoded_smi = np.array(encoded_smi)
    return encoded_smi[np.newaxis, :]
