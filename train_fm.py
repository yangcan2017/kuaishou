#coding=utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from pandas import get_dummies
from sklearn.model_selection import StratifiedKFold
import gc

def pandas_onehot(df, col):
    df = get_dummies(df, columns=col)
    return df

def sklearn_onehoot(df):
    enc = OneHotEncoder()
    enc.fit(df)
    data = enc.transform(df.drop(['label'], axis=1)).toarray()
    return data

def data_convert2ffm(df, ffmfile):
    columns = df.columns.values
    d = len(columns)
    print d
    feature_index = [i for i in range(d)]  #默认从0开始
    print feature_index
    field_index = [0]*d #初始化参数
    field = [] #初始化参数
    for col in columns:
        field.append(col.split('_')[0])  #onehot选出编码前的变量
    index = -1
    for i in range(d):
        if i==0 or field[i]!=field[i-1]:  #判断是否在同一个field里面
            index += 1
        field_index[i] = index           #默认从0开始

    with open(ffmfile, 'w') as f:
        for row in df.values:
            print row
            line = str(row[0])  #label
            for i in range(1, len(row)):
                if row[i]!=0:
                    line += ' ' + "%d:%d:%d" % (field_index[i], feature_index[i], row[i]) + ' '
            line += '\n'
            f.write(line)

    print('finishing......')

train_inte = pd.read_csv('input/train_interaction.txt', sep='\t', header=None)
train_inte.columns = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
df_train = train_inte[['click', 'user_id', 'photo_id', 'time',  'duration_time']]
df_trainoo = pandas_onehot(df_train, ['user_id', 'photo_id', 'time',  'duration_time'])
df_trainoo['click'] = df_train['click']
data_convert2ffm(df_trainoo, './input/train.ffm')



# test_inte = pd.read_csv('input/test_interaction.txt', sep='\t', header=None)
# test_inte.columns = ['user_id', 'photo_id', 'time', 'duration_time']

