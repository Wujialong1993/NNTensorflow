#先生成一行数据代表一列，每一列正态分布随机生成，再每一列归一化处理，再把其映射到标准范围
import numpy as np
import csv


row=3
col=6
#一共六个参数,每一行的第一个数代表这个参数的最大值,第二个数代表最小值
maxAndMinValue=[[23.47, 15.43],
                [28.04, 16.78],
                [42.10, 27.90],
                [74.20, 49.60],
                [31.81, 20.66],
                [20.69, 13.50]]
maxAndMinValue=np.array(maxAndMinValue)

#inputData
inputData=np.ones((col, row))
for i in range(0,col,1):
    data_value = np.random.normal(0, 1, size=(row))
    maxValue=np.max(data_value)
    minValue=np.min(data_value)
    for j in range(0,row,1):
        data_value[j]= maxAndMinValue[i][1]+((data_value[j] - minValue) / (maxValue - minValue))*(maxAndMinValue[i][0]-maxAndMinValue[i][1])
    inputData[i]=data_value
#print(inputData)
inputData=np.transpose(inputData)
# print(inputData)
# print(maxAndMinValue)
print(inputData)

for i in range (0,len(inputData)):
   for j in range(0,len(inputData[i])):
       inputData[i][j]=0.01;
print(inputData)
