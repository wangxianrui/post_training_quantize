import tensorflow as tf
import numpy as np

a = np.random.randint(0, 10, [3, 4], np.uint8)
a = a.astype(np.float)
print(a)
print(a - 10)
