import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_process_utils import create_folder
from src.CONSTS import MOL_DICT, MAX_MOL_LEN, BATCH_SIZE


def get_train_val_test_data():
    create_folder('data/train_data/')
    create_folder('data/test_data/')
    df_data_1 = pd.read_csv('data/data_clean_a2d.csv')
    df_data_2 = pd.read_csv('data/data_clean_kop.csv')
    breakpoint()

    # train, val, test split
    df_train, df_test \
        = train_test_split(df_data, test_size=0.05, random_state=43)

    df_train, df_val \
        = train_test_split(df_train, test_size=0.01, random_state=43)

    df_train.to_csv('data/train_data/df_train.csv', index=False)
    df_test.to_csv('data/test_data/df_test.csv', index=False)
    df_val.to_csv('data/test_data/df_val.csv', index=False)


def get_encoded_smi(smi):
    encoded_smi = []
    for char in smi:
        encoded_smi.append(MOL_DICT.index(char))

    if len(encoded_smi) <= MAX_MOL_LEN:
        num_pads = MAX_MOL_LEN - len(encoded_smi)
        # 39 is the padding number which will be masked
        encoded_smi += [len(MOL_DICT)] * num_pads
    else:
        encoded_smi = encoded_smi[:MAX_MOL_LEN]
    return encoded_smi


def get_val_data():
    df_val = pd.read_csv('data/test_data/df_val.csv')
    x = []
    y = []
    for _, row in df_val.iterrows():
        x.append(get_encoded_smi(row.X))
        y.append(get_encoded_smi(row.Y))

    _data = (np.vstack(x), np.vstack(y))
    with open('data/test_data/' + 'Xy_val.pkl', 'wb') as f:
        pickle.dump(_data, f)


def data_iterator_train():
    df_train = pd.read_csv('data/train_data/df_train.csv')
    while True:
        df = df_train.sample(frac=1).reset_index(drop=True)
        x = []
        y = []
        for _, row in df.iterrows():
            x.append(get_encoded_smi(row.X))
            y.append(get_encoded_smi(row.Y))
            if len(x) >= BATCH_SIZE:
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
        if len(x) >= BATCH_SIZE:
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
    df_train = pd.read_csv('data/train_data/df_train.csv')
    for x, y in data_iterator_train():
        print(x.shape)
