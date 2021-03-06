import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_process_utils import create_folder, standardize_smi, get_encoded_smi
from src.CONSTS import BATCH_SIZE_PRED


def get_train_val_test_data():
    create_folder('predictor_data/train_data/')
    create_folder('predictor_data/test_data/')
    df_data_1 = pd.read_csv('predictor_data/data_clean_a2d.csv')
    df_data_2 = pd.read_csv('predictor_data/data_clean_kop.csv')
    df_data = pd.concat([df_data_1, df_data_2])
    df_data.loc[:, 'Data'] = df_data.SMILES.map(standardize_smi)

    # train, val, test split
    df_train, df_test \
        = train_test_split(df_data, test_size=0.1, random_state=43)

    df_train, df_val \
        = train_test_split(df_train, test_size=0.1, random_state=43)

    df_train.to_csv('predictor_data/train_data/df_train.csv', index=False)
    df_test.to_csv('predictor_data/test_data/df_test.csv', index=False)
    df_val.to_csv('predictor_data/test_data/df_val.csv', index=False)
    max_y = np.quantile(df_train.pCHEMBL.values, 0.98)
    min_y = np.quantile(df_train.pCHEMBL.values, 0.02)
    with open('predictor_data/train_data/' + 'y_max_min.pkl', 'wb') as f:
        pickle.dump((min_y, max_y), f)


def get_val_data():
    df_val = pd.read_csv('predictor_data/test_data/df_val.csv')
    with open('predictor_data/train_data/y_max_min.pkl', 'rb') as handle:
        y_min, y_max = pickle.load(handle)
    x = []
    y = []
    for _, row in df_val.iterrows():
        x.append(get_encoded_smi(row.Data))
        _y = (row.pCHEMBL - y_min) / (y_max - y_min)
        y.append(_y)

    _data = (np.vstack(x), np.vstack(y))
    with open('predictor_data/test_data/' + 'Xy_val.pkl', 'wb') as f:
        pickle.dump(_data, f)


def data_iterator_train():
    df_train = pd.read_csv('predictor_data/train_data/df_train.csv')
    with open('predictor_data/train_data/y_max_min.pkl', 'rb') as handle:
        y_min, y_max = pickle.load(handle)
    while True:
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        x = []
        y = []
        for _, row in df_train.iterrows():
            x.append(get_encoded_smi(row.Data))
            _y = (row.pCHEMBL - y_min) / (y_max - y_min)
            y.append(_y)
            if len(x) >= BATCH_SIZE_PRED:
                yield (np.vstack(x), np.vstack(y))
                x = []
                y = []

        if x:
            yield (np.vstack(x), np.vstack(y))
            x = []
            y = []


def data_iterator_test(test_df_path):
    df_test = pd.read_csv(test_df_path)
    with open('predictor_data/train_data/y_max_min.pkl', 'rb') as handle:
        y_min, y_max = pickle.load(handle)
    x = []
    y = []
    for _, row in df_test.iterrows():
        x.append(get_encoded_smi(row.Data))
        _y = (row.pCHEMBL - y_min) / (y_max - y_min)
        y.append(_y)
        if len(x) >= BATCH_SIZE_PRED:
            yield (np.vstack(x), np.vstack(y))
            x = []
            y = []

    if x:
        yield (np.vstack(x), np.vstack(y))
        x = []
        y = []


if __name__ == "__main__":
    get_train_val_test_data()
    get_val_data()
    # df_train = pd.read_csv('predictor_data/train_data/df_train.csv')
    # for x, y in data_iterator_test('predictor_data/test_data/df_test.csv'):
    #     breakpoint()
    #     print(x.shape)
    # # for x, y in data_iterator_train():
    # #     print(x.shape)
