EMBEDDING_SIZE = 256
FFD_SIZE = 256
MAX_MOL_LEN = 65
NUM_HEADS = 2
NUM_LAYERS = 4
DROPOUT_RATE = 0.1
BATCH_SIZE = 64

atoms = [
    'H', 'B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I',
    '[Se]', '[Na+]', '[Si]'
]
special = [
    '(', ')', '[', ']', '=', '#', '@', '*', '%', '0', '1', '2',
    '3', '4', '5', '6', '7', '8', '9', '.', '/', '\\', '+', '-',
    'c', 'n', 'o', 's', 'p'
]
padding = ['A', 'E']  # Go, End


MOL_DICT = sorted(atoms, key=len, reverse=True) + special + padding
