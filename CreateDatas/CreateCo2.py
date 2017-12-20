#先生成一行数据代表一列，每一列正态分布随机生成，再每一列归一化处理，再把其映射到标准范围
import numpy as np
import csv
np.set_printoptions(threshold=np.inf)
csv_reader_Input= csv.reader(open('../datas/csvInputData.csv', encoding='utf-8'))
#从csv文件读取输入数据
#返回值res为一个list
def readInputeDataAsListFromCsv(csv_reader):
    res=[]
    for row in csv_reader:
        myRow = []
        for i in range(0, len(row)):
            myRow.append(float(row[i]))
        res.append(myRow)
    tjmd = np.array(res)
    return res

#input 为输入的二维数组（n*6）
#ws是一维数组，为系数矩阵（1*6）
#返回值label是二维数组（n*6）
def generateLabel(input,ws):
    label=np.ones((len(inputDatas), 1))
    for i in range(0, len(inputDatas), 1):
        value = ws[0] * inputDatas[i][0] + ws[1] * inputDatas[i][1] + ws[2] * inputDatas[i][2] + ws[3] * inputDatas[i][3] + ws[4] * inputDatas[i][4] + ws[5] * inputDatas[i][5]
        label[i][0] = value
    return label
#该函数将label归一并添加噪声
#maxValue为label所允许的最大值
#minValue为label所允许的最小值
#noiseRate为噪声率，噪声服从正态分布,其中μ=noiseRate*(maxValue-minValue)
def labelNormalizeAndNoise(label,maxValue,minValue,noiseRate):
    maxLabel = np.max(label)
    minLabel = np.min(label)
    for i in range(0, len(label), 1):
        temp = (label[i][0] - minLabel) / (maxLabel - minLabel)
        label[i][0] = minValue + temp * (maxValue - minValue)
    avg=np.average(label)
    u=noiseRate*(maxValue-minValue)
    noise = np.random.normal(0, u, label.shape)
    label = label + noise
    #print(np.max(noise))
    return label
#该函数是将总体数据分层，分出训练数据与测试数据
#data为待分数据集
#trainRate 为训练数据所占比重
#返回值为二维数组，先返回训练集，再返回测试集
def separateDataAsTrainAndTest(data,trainRate):
    train = []
    test = []
    for i in range(0, len(data)):
        if (i < 0.7 * len(data)):
            train.append(data[i])
        else:
            test.append(data[i])
    train = np.array(train)
    test = np.array(test)
    return train,test

#该方法是将生成的数据保存到csv文件中
#datas为需要存到csv文件中的数据源
#path为csv文件的保存路径
def saveToCsv(datas,path):
    csv_File = open(path, 'w', newline='')
    csv_writer = csv.writer(csv_File)
    csv_writer.writerows(datas)
#Co2 Label
inputDatas=readInputeDataAsListFromCsv(csv_reader_Input)
inputDatas=np.array(inputDatas)
ws=[0.4,0.6,0.7,0.8,0.5,0.9]
ws=np.array(ws)
label=generateLabel(inputDatas,ws)
label=labelNormalizeAndNoise(label,1.60,1.53,0.1)#=================================
data =np.concatenate((inputDatas, label),axis=1)
print(data)
train,test=separateDataAsTrainAndTest(data,0.7)
print(train.shape)
print(test.shape)
saveToCsv(data,'../datas/csvCo2_All.csv')
saveToCsv(train,'../datas/TrainData/csvCo2_Train.csv')
saveToCsv(test,'../datas/TestData/csvCo2_Test.csv')
