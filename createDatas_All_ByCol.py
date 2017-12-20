import numpy as np
s = np.random.normal(0, 1, size=(3,6))
print(s)

maxValue=np.max(s)
minValue=np.min(s)
def get_max_min_value(martix):
    '''''
    得到矩阵中每一列最大的值
    '''
    max_list=[]
    min_list=[]
    for j in range(len(martix[0])):
        one_list=[]
        for i in range(len(martix)):
            one_list.append(float(martix[i][j]))
        max_list.append(float(max(one_list)))
        min_list.append(float(min(one_list)))
    return max_list,min_list
def max_min_normalization(data_value, data_col_max_values, data_col_min_values):
    """ Data normalization using max value and min value

    Args:
        data_value: The data to be normalized
        data_col_max_values: The maximum value of data's columns
        data_col_min_values: The minimum value of data's columns
    """
    data_shape = data_value.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]

    for i in range(0, data_rows, 1):
        for j in range(0, data_cols, 1):
            data_value[i][j] = (data_value[i][j] - data_col_min_values[j]) / (data_col_max_values[j] - data_col_min_values[j])

if __name__ == '__main__':
    martix=[['1','2','3'],['3','5','0'],['5','6','2']]
    maxValue,minValue=get_max_min_value(s)
    print(maxValue)
    print(minValue)
    max_min_normalization(s,maxValue,minValue)
    print(s)
#
# print(s)

