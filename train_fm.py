#coding=utf-8

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder
from pandas import get_dummies
from sklearn.model_selection import StratifiedKFold
import gc

# def pandas_onehot(df, col):
#     df = get_dummies(df, columns=col)
#     return df
#
# def sklearn_onehoot(df):
#     enc = OneHotEncoder()
#     enc.fit(df)
#     data = enc.transform(df).tocsr()
#     return data

def data_convert2ffm(df, field, col):
    enc = OneHotEncoder()
    enc.fit(df)
    data = enc.transform(df).tocsr()
    feature_index = []
    value = []
    data_ffm={}
    # for d in data:
    #     feature_index.append(str(d).split('\t')[0].strip().replace('(', '').replace(')', '').split(',')[1].strip())
    #     value.append(str(d).split('\t')[1])

    data_ffm[col] = ["%d:%d:%f" % (field,
        int(str(d).split('\t')[0].strip().replace('(', '').replace(')', '').split(',')[1].strip()),
        float(str(d).split('\t')[1])) for d in data]
    df_data = DataFrame(data_ffm, columns=[col])
    df_data.to_csv('output/' + col + '.csv', sep=',', index=False, header=False)

    # columns = df.columns.values
    # d = len(columns)
    # feature_index = [i for i in range(d)]  #默认从0开始
    # field_index = [0]*d #初始化参数
    # field = [] #初始化参数
    # for col in columns:
    #     field.append(col.split('_')[0])  #onehot选出编码前的变量
    # index = -1
    # for i in range(d):
    #     if i==0 or field[i]!=field[i-1]:  #判断是否在同一个field里面
    #         index += 1
    #     field_index[i] = index           #默认从0开始
    #
    # with open(ffmfile, 'w') as f:
    #     for row in df.values:
    #         line = str(row[0])  #label
    #         for i in range(1, len(row)):
    #             if row[i]!=0:
    #                 line += ' ' + "%d:%d:%d" % (field_index[i], feature_index[i], row[i]) + ' '
    #         line += '\n'
    #         f.write(line)
    #
    # print('finishing......')

train_inte = pd.read_csv('input/train_interaction.txt', sep='\t', header=None)
train_inte.columns = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
for col in ['user_id', 'photo_id', 'time', 'duration_time']:
    print col
    data_convert2ffm(train_inte[[col]], 1, col)

# df_photo_id = sklearn_onehoot(train_inte[['photo_id']])
# df_dura_time = sklearn_onehoot(train_inte[['duration_time']])

# df_trainoo['click'] = df_train['click']
# data_convert2ffm(df_train, './input/train.ffm')



# test_inte = pd.read_csv('input/test_interaction.txt', sep='\t', header=None)
# test_inte.columns = ['user_id', 'photo_id', 'time', 'duration_time']

