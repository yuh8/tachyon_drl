MAX_MOL_LEN = 65
BATCH_SIZE = 16

atoms = [
    'H', 'B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I',
    '[Se]', '[Na+]', '[Si]'
]
special = [
    '(', ')', '[', ']', '=', '#', '@', '*', '%', '0', '1', '2',
    '3', '4', '5', '6', '7', '8', '9', '.', '/', '\\', '+', '-',
    'c', 'n', 'o', 's', 'p'
]
padding = ['A', 'E']  # Go, Padding ,End


MOL_DICT = sorted(atoms, key=len, reverse=True) + special + padding
ALPHA = 0.01
