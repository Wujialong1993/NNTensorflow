#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import tensorflow as tf
import numpy as np

# 定义两个矩阵
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

matrix3 = tf.constant(3,shape=(1,2))
matrix4 = tf.constant(2,shape=(2,1))
# 定义矩阵乘法
product = tf.matmul(matrix1, matrix2)

# 运行矩阵乘法，session用法一
sess = tf.Session()
result = sess.run(product)
print ('Session用法一')
print (result)
print (matrix1)
print (matrix2)
print (matrix3)
print (matrix4)
sess.close()

## session用法二，不用考虑close，会自动关闭

with tf.Session() as sess:
    result = sess.run(product)
    print ('Session用法二')
    print (result)