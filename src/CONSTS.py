import pickle

EMBEDDING_SIZE_GEN = 512
FFD_SIZE_GEN = 512
EMBEDDING_SIZE_PRED = 64
FFD_SIZE_PRED = 64
NUM_HEADS_GEN = 4
NUM_LAYERS_GEN = 6
NUM_HEADS_PRED = 1
NUM_LAYERS_PRED = 2
BATCH_SIZE_GEN = 64
BATCH_SIZE_PRED = 16
BATCH_SIZE_RL = 16
MAX_MOL_LEN = 65
DROPOUT_RATE = 0.1
ALPHA = 0.01


atoms = [
    'H', 'B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I',
    '[Se]', '[Na+]', '[Si]'
]
special = [
    '(', ')', '[', ']', '=', '#', '@', '*', '%', '0', '1', '2',
    '3', '4', '5', '6', '7', '8', '9', '.', '/', '\\', '+', '-',
    'c', 'n', 'o', 's', 'p'
]
padding = ['G', 'E']  # Go, End


MOL_DICT = sorted(atoms, key=len, reverse=True) + special + padding


with open('predictor_data/train_data/y_max_min.pkl', 'rb') as handle:
    Y_MIN, Y_MAX = pickle.load(handle)
