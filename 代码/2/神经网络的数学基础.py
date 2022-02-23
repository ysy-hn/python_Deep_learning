import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示warning和error


# from keras.datasets import mnist
# from keras import models,layers,regularizers
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.utils import to_categorical
#
#
# # 加载数据集
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#
# # 展平
# train_images = train_images.reshape((60000, 28*28)).astype('float')
# test_images = test_images.reshape((10000, 28*28)).astype('float')
#
# # 将标签以onehot编码
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
#
# # 建立模型
# network = models.Sequential()
# # kernel_regularizer正则化降低过拟合
# network.add(layers.Dense(units=128, activation='relu', input_shape=(28*28, ), kernel_regularizer=regularizers.l1(0.0001)))
# # drout层降低过拟合
# network.add(layers.Dropout(0.01))
# network.add(layers.Dense(units=32, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
# network.add(layers.Dropout(0.01))
# network.add(layers.Dense(units=10, activation='softmax'))
#
# # 编译
# network.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# # 训练
# network.fit(train_images, train_labels, epochs=20, batch_size=128, verbose=2)
#
# # 在测试集上测试模型性能
# test_loss, test_accuracy = network.evaluate(test_images, test_labels)
# print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)


# from keras.datasets import mnist
# import matplotlib.pyplot as plt
#
#
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#
# print(train_images.ndim)
# print(train_images.shape)
# print(train_images.dtype)
#
# digit = train_images[4]
# plt.imshow(digit, cmap=plt.cm.binary)  # imshow:灰意度
# plt.show()
#
# my_slcie = train_images[10:100]
# print(my_slcie.shape)


# def naive_relu(x):
#     assert len(x.shape) == 2
#
#     x = x.copy()
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             x[i, j] = max(x[i, j], 0)
#     return x
#
# def naive_add(x, y):
#     assert len(x.shape) == 2
#     assert x.shape == y.shape
#
#     x = x.copy()
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             x[i, j] += y[i, j]
#     return x
#
#
# import numpy as np
#
#
# x = np.array([[0, 1], [1, 1]])
# y = np.array([[0, 1], [1, 2]])
# z = x + y
# z = np.maximum(z, 1.)
# print(z)
# print(naive_relu(x))
# print(naive_add(x, y))

# def naive_add_matrix_and_vector(x, y):
#     assert len(x.shape) == 2
#     assert len(y.shape) == 1
#     assert x.shape[1] == y.shape[0]
#
#     x = x.copy()
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             x[i, j] += y[j]
#     return x
#
# import numpy as np
#
# x = np.random.random((64, 3, 32, 10))
# y = np.random.random(((32, 10)))
# z = np.maximum(x, y)

# # 2.3 张量点积
# def naive_vector_dot(x, y):
#     assert len(x.shape) == 1
#     assert len(y.shape) == 1
#     assert x.shape[0] == y.shape[0]
#
#     z = 0.
#     for i in range(x.shape[0]):
#         z += x[i] * y[i]
#     return z

