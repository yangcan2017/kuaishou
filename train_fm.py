#coding=utf-8

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder
from pandas import get_dummies
from sklearn.model_selection import StratifiedKFold
import gc
import xlearn as xl


def data_convert2ffm(df, field, col):
    enc = OneHotEncoder()
    enc.fit(df)
    data = enc.transform(df).tocsr()
    data_ffm={}
    data_ffm[col] = ["%d:%d:%f" % (field,
        int(str(d).split('\t')[0].strip().replace('(', '').replace(')', '').split(',')[1].strip()),
        float(str(d).split('\t')[1])) for d in data]
    df_data = DataFrame(data_ffm, columns=[col])
    df_data.to_csv('output/' + col + '_test.csv', sep=',', index=False, header=False)


ffm_model = xl.create_ffm()
# ffm_model.setTrain("./output/train20180606.ffm")

# param = {'task':'binary', 'k':2, 'lr':0.1, 'lambda':0.0002, 'metric':'auc', 'epoch':25}
# ffm_model.fit(param, "model.out")
ffm_model.setTest("./output/test20180606.ffm")
ffm_model.predict("model.out", "output20180606.txt")