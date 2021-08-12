import glob
import pandas as pd
from .data_process_utils import get_df_from_smi

EMBEDDING_SIZE = 128
FFD_SIZE = 128
MAX_MOL_LEN = 65
NUM_HEADS = 8
NUM_LAYERS = 4
BATCH_SIZE = 64
MOL_DICT = '#%()+-/0123456789=ABCEFHINOPS[\\]clnoprs'

if not glob.glob("data/df_data.csv"):
    get_df_from_smi("data/data_train.smi")

df_train = pd.read_csv("data/df_data.csv")
all_tokens = df_train.Data.values.tolist()
all_tokens = "".join(all_tokens)

if len(set(all_tokens)) > len(MOL_DICT):
    MOL_DICT = ''.join(set(all_tokens))
    MOL_DICT = sorted(MOL_DICT)
    MOL_DICT = ''.join(MOL_DICT)
