#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

# 随机生成100个浮点数
x_data = np.random.rand(10000).astype(np.float32)
# y=0.1*x+0.3，这样就构造了训练数据(x_data,y_data)
y_data = x_data*x_data*x_data*x_data * 0.1 + 0.9

# 定义线性回归的权重参数
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# 定义线性回归的偏置参数
Biases = tf.Variable(tf.zeros([1]))

# 构建线性回归模型
y = Weights * x_data + Biases
# 计算线性回归的损失函数
loss = tf.reduce_mean(tf.square(y - y_data))

# 定义线性回归的求解方法，梯度下降法，学习率为0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 使用tensorflow求解
train = optimizer.minimize(loss)
# 初始化tensorflow的所有变量
init = tf.global_variables_initializer()
# 定义tensorflow的session
sess = tf.Session()
# 将初始化数据放入到session中，执行时会用到
sess.run(init)

# 迭代201次求解线性回归参数
for step in range(2000):
    # session执行训练
    sess.run(train)
    if step % 20 == 0:
        # 每迭代二十次输出一次结果
        print (step, sess.run(Weights), sess.run(Biases))