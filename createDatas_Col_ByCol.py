#先生成一行数据代表一列，每一列正态分布随机生成，再每一列归一化处理，再把其映射到标准范围
import numpy as np
import csv


row=1000
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



#lable
tjmds=np.ones((row, 1))
ws=[0.4,0.6,0.7,0.8,0.5,0.9]
ws=np.array(ws)
for i in range(0,row,1):
    tjmd=ws[0]*inputData[i][0] + ws[1]*inputData[i][1] + ws[2]*inputData[i][2] + ws[3]*inputData[i][3] + ws[4]*inputData[i][4] + ws[5]*inputData[i][5]
    tjmds[i][0]=tjmd
#tjmds=np.array(tjmds)



maxtjmd=np.max(tjmds)
mintjmd=np.min(tjmds)
for i in range(0,row,1):
    temp = (tjmds[i]-mintjmd)/(maxtjmd-mintjmd)
    #print("temp:", temp)
    tjmds[i] = 1.52 + temp * (1.59-1.52)
    #print(tjmds[i])
#print(tjmds)
#tjmds=np.transpose(tjmds)
#print(tjmds)
# print(tjmds.shape)
# print(inputData.shape)

# for i in range(0,col,1):
#     data_value = np.random.normal(0, 1, size=(row))
#     maxValue=np.max(data_value)
#     minValue=np.min(data_value)
#     for j in range(0,row,1):


# data =np.concatenate((inputData, tjmds),axis=1)
# print(data)
#
#
# csv_File=open('csvTjmdData.csv','w',newline='')
# csv_writer=csv.writer(csv_File)
# csv_writer.writerows(data)








