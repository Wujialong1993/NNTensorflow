#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
csv_reader_Train = csv.reader(open('../datas/TrainData/csvTjmd_Train.csv', encoding='utf-8'))
csv_reader_Test = csv.reader(open('../datas/TestData/csvTjmd_Test.csv', encoding='utf-8'))

# 创建一个神经网络层
def add_layer(input, in_size, out_size, activation_function = None):
    """
    :param input:
        神经网络层的输入
    :param in_zize:
        输入数据的大小
    :param out_size:
        输出数据的大小
    :param activation_function:
        神经网络激活函数，默认没有
    """

    # 定义神经网络的初始化权重
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 定义神经网络的偏置
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.5)
    # 计算w*x+b
    W_mul_x_plus_b = tf.matmul(input, Weights) + biases
    # 根据是否有激活函数进行处理
    if activation_function is None:
        output = W_mul_x_plus_b
    else:
        output = activation_function(W_mul_x_plus_b)

    return output

#该方法是将生成的数据保存到csv文件中
#datas为需要存到csv文件中的数据源
#path为csv文件的保存路径
def saveToCsv(datas,path):
    csv_File = open(path, 'w', newline='')
    csv_writer = csv.writer(csv_File)
    csv_writer.writerows(datas)

#数据归一化函数
def normalize(datas):
    res=datas.transpose()
    for i in range(0, len(res)):
        max = np.max(res[i])
        min = np.min(res[i])
        for j in range(0, len(res[i])):
            res[i][j] = (res[i][j] - min) / (max - min)
    res=res.transpose()
    return res

#从文件中读取数据，col为label列索引
def getDatasFromCsv(csv_reader, col):
    data = []
    label = []
    for row in csv_reader:
        myRow = []
        myhf = []
        for i in range(0, 6):
            myRow.append(float(row[i]))
        myhf.append(float(row[col]))
        data.append(myRow)
        label.append(myhf)
    data = np.array(data)
    label = np.array(label)
    return data,label

#该方法是预测完成后，用折线图的方式显示预测值和真实值
#datas为二维数组，数组第一列：【真实数据】 第二列：【预测数据】
#对应图中显示，蓝色线条：【真实数据】 红色线条：【预测数据】
def showPridictLineChartByPlt(datas):
    x = np.linspace(0, len(datas), num=len(datas))
    y1_actual = np.zeros_like(x, float)
    y2_pridict = np.zeros_like(x, float)
    for i in range(0, len(datas)):
        y1_actual[i] = datas[i][0]
        y2_pridict[i] = datas[i][1]
    plt.figure()
    plt.plot(x, y1_actual)
    plt.plot(x, y2_pridict, color='red', linewidth=1.0, linestyle='-')
    max=np.max(datas)
    min=np.min(datas)
    # 设置坐标轴的取值范围
    # plt.xlim((0, len(datas)))
    # plt.ylim((min, max))
    plt.show()

    # 设置坐标轴的lable
    # plt.xlabel('X axis')
    # plt.ylabel('Y axis')


#准备训练数据
data_train,label_train=getDatasFromCsv(csv_reader_Train,6)

#训练数据归一化
data_train=normalize(data_train)

#test datas
data_test,label_test=getDatasFromCsv(csv_reader_Test,6)
#归一化测试数据
data_test=normalize(data_test)

# print(data_test)

# 定义输入数据，None是样本数目，表示多少输入数据都行，1是输入数据的特征数目
xs = tf.placeholder(tf.float32, [1, 6])
# 定义输出数据，与xs同理
ys = tf.placeholder(tf.float32, [1, 1])
# 定义一个隐藏层
hidden_layer1 = add_layer(xs, 6, 12, activation_function = tf.nn.sigmoid)
hidden_layer2 = add_layer(hidden_layer1, 12, 18, activation_function = tf.nn.sigmoid)
hidden_layer3 = add_layer(hidden_layer2, 18, 9, activation_function = tf.nn.sigmoid)
# 定义输出层
prediction = add_layer(hidden_layer3, 9, 1, activation_function = None)

# 求解神经网络参数
# 定义损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices = [1]))
# 定义训练过程
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
# 变量初始化
init = tf.global_variables_initializer()
# 定义Session
sess = tf.Session()
# 执行初始化工作
sess.run(init)

# 进行训练
#一条数据执行一次，总共700条训练数据，重复执行训练30次
for j in range(30):
    for i in range(700):
        # 执行训练，并传入数据
        x_data=[]
        x_data.append(data_train[i])
        x_data=np.array(x_data)
        y_data = []
        y_data.append(label_train[i])
        y_data = np.array(y_data)
        sess.run(train_step, feed_dict = {xs: x_data, ys: y_data})
        if i % 100 == 0:
           print (sess.run(loss, feed_dict = {xs: x_data, ys: y_data}))

#预测数据，一条预测一次，一共300条预测数据，预测300次
res=[]
for i in range(300):
    x_data = []
    x_data.append(data_test[i])
    x_data = np.array(x_data)
    print(x_data ,sess.run(prediction, feed_dict = {xs: x_data}))
    res.append(sess.run(prediction, feed_dict={xs: x_data})[0])
res=np.array(res)
res =np.concatenate((label_test, res),axis=1)
print(res)

#保存数据到csv data=[真实值,预测值]
saveToCsv(res,'../datas/PredictData/csvTjmd_Predict.csv')
#以折线图的形式显示预测数据和真实数据：蓝色-【真实数据】；红色-【预测数据】
showPridictLineChartByPlt(res)
# 关闭Session
sess.close()