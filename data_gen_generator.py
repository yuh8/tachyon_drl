import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_process_utils import create_folder, get_df_from_smi, get_encoded_smi
from src.CONSTS import BATCH_SIZE_GEN


def get_train_val_test_data():
    df_data = get_df_from_smi('generator_data/data_train.smi')
    create_folder('generator_data/train_data/')
    create_folder('generator_data/test_data/')

    # train, val, test split
    df_train, df_test \
        = train_test_split(df_data, test_size=0.05, random_state=43)

    df_train, df_val \
        = train_test_split(df_train, test_size=0.01, random_state=43)

    df_train.to_csv('generator_data/train_data/df_train.csv', index=False)
    df_test.to_csv('generator_data/test_data/df_test.csv', index=False)
    df_val.to_csv('generator_data/test_data/df_val.csv', index=False)


def get_val_data():
    df_val = pd.read_csv('generator_data/test_data/df_val.csv')
    x = []
    y = []
    for _, row in df_val.iterrows():
        x.append(get_encoded_smi(row.X))
        y.append(get_encoded_smi(row.Y))

    _data = (np.vstack(x), np.vstack(y))
    with open('generator_data/test_data/' + 'Xy_val.pkl', 'wb') as f:
        pickle.dump(_data, f)


def data_iterator_train():
    df_train = pd.read_csv('generator_data/train_data/df_train.csv')
    while True:
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        x = []
        y = []
        for _, row in df_train.iterrows():
            x.append(get_encoded_smi(row.X))
            y.append(get_encoded_smi(row.Y))
            if len(x) >= BATCH_SIZE_GEN:
                yield (np.vstack(x), np.vstack(y))
                x = []
                y = []

        if x:
            yield (np.vstack(x), np.vstack(y))
            x = []
            y = []


def data_iterator_test(test_df_path):
    df_test = pd.read_csv(test_df_path)
    x = []
    y = []
    for _, row in df_test.iterrows():
        x.append(get_encoded_smi(row.X))
        y.append(get_encoded_smi(row.Y))
        if len(x) >= BATCH_SIZE_GEN:
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
    # df_train = pd.read_csv('generator_data/train_data/df_train.csv')
    # for x, y in data_iterator_train():
    #     print(x.shape)
