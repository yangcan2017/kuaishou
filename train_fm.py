#coding=utf-8

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder
from pandas import get_dummies
from sklearn.model_selection import StratifiedKFold
import gc
import xlearn as xl


ffm_model = xl.create_ffm()
ffm_model.setTrain("./input/train20180607.ffm")
ffm_model.setValidate("./input/va20180607.ffm")

param = {'task':'binary', 'k':2, 'lr':0.1, 'lambda':0.0002, 'metric':'auc', 'epoch':25}
ffm_model.fit(param, "model.out")

#test
ffm_model.setTest("./output/test20180606.ffm")
ffm_model.setSigmoid()
ffm_model.predict("model.out", "output20180606.txt")
