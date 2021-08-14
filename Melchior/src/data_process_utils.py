import os
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
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
