import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


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
            # A and E are speical tokens are start and end, respectively
            standardized_smi = "G" + parse_mol_to_smi(mol) + "E"
            x.append(standardized_smi[:-1])
            y.append(standardized_smi[1:])
            data.append(standardized_smi)

    df = pd.DataFrame()
    df.loc[:, 'X'] = x
    df.loc[:, 'Y'] = y
    df.loc[:, 'Data'] = data
    df.to_csv('../data/df_data.csv', index=False)


if __name__ == '__main__':
    get_df_from_smi("../data/data_train.smi")
