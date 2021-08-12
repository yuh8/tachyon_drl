import tensorflow as tf
from src.CONSTS import MOL_DICT
print(tf.where([True, False, False, True], [1, 2, 3, 4], [100, 200, 300, 400]))
print(tf.linalg.band_part(tf.ones((3, 3)), -1, 0))
