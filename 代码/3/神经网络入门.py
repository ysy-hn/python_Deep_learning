# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示warning和error

from keras import layers

layer = layers.Dense(32, input_shape=(784,))
