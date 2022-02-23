import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 默认，显示所有信息。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示warning和error
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示error


import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300
# dot：主要用于矩阵的乘法运算，其中包括：向量内积、多维矩阵乘法和矩阵与向量的乘法。

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()
# tf.initialize_all_variables()已修改为上述。

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))


import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

product = tf.matmul(matrix1, matrix2)

# 在一个会话中启动图
sess = tf.Session()

result = sess.run(product)
print(result)
sess.close()

with tf.Session() as sess:
    result = sess.run(product)
    print(result)
# 正常启动图后需要关闭，可使用类似打开文件操作。

# 主动分配GPU参与计算
with tf.Session() as sess:
    with tf.device("/gpu:1"):
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.], [2.]])
        product = tf.matmul(matrix1, matrix2)
        result = sess.run(product)
        print(result)
# with...Device 语句用来指派特定的 CPU 或 GPU 执行操作:
# "/cpu:0": 机器的 CPU.
# "/gpu:0": 机器的第一个 GPU, 如果有的话.
# "/gpu:1": 机器的第二个 GPU, 以此类推.

# 交互式使用
# InteractiveSession代替Session类,使用Tensor.eval()和Operation.run()方法
# 代替Session.run(). 这样可以避免使用一个变量来持有会话.
import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

x.initializer.run()

sub = tf.subtract(x, a)  # subtract：减法
print(sub.eval())


# 变量
import tensorflow as tf

state = tf.Variable(0, name="counter")  # 创建一个变量, 初始化为标量 0.

one = tf.constant(1)  # 创建一个 op, 其作用是使 state 增加 1
new_value = tf.add(state, one)
update = tf.assign(state, new_value)  # assign：将new_calue分配给state

init_op = tf.global_variables_initializer()  # 启动图后, 变量必须先经过`初始化` (init) op 初始化,

with tf.Session() as sess:  # 启动图, 运行 op
  sess.run(init_op)    # 运行 'init' op
  print(sess.run(state))    # 打印 'state' 的初始值
  for _ in range(3):    # 运行 op, 更新 'state', 并打印 'state'
      sess.run(update)
      print(sess.run(state))

# fetch（获取）
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)  # multiply:矩阵对应元素相乘。

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)

# feed
# 临时替代图中的任意操作中的 tensor 可以对图中任何操作提交补丁, 直接插入一个 tensor.
# 最常见的用例是将某些特殊的操作指定为 "feed" 操作,
# 标记的方法是使用 tf.placeholder() 为这些操作创建占位符.
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
# placeholder 是 Tensorflow 中的占位符，暂时储存变量.
# Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(),
# 然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).
# 需要传入的值放在了feed_dict={} 并一一对应每一个 input.
# placeholder 与 feed_dict={} 是绑定在一起出现的。

# 一般运算
# add：加法；subtract：减法。
# matmul：矩阵乘法；multiply:矩阵对应元素相乘。
# dot：主要用于矩阵的乘法运算，其中包括：向量内积、多维矩阵乘法和矩阵与向量的乘法。
# assign（1， 2）：将后者2分配给前者1。
