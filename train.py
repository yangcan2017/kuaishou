#coding=utf-8

import pandas as pd
import numpy as np
from scipy import sparse as ssp
import gc

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

np.random.seed(0)

train_inte = pd.read_csv('input/train_interaction.txt', sep='\t', header=None)
train_inte.columns = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']

test_inte = pd.read_csv('input/test_interaction.txt', sep='\t', header=None)
test_inte.columns = ['user_id', 'photo_id', 'time', 'duration_time']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(train_inte['user_id'], train_inte['click'])
train_oov = np.zeros(train_inte.shape[0])
test_oov = np.zeros(test_inte.shape[0])

for ind_tr, ind_te in skf:
    data_tr = train_inte.iloc[ind_tr]
    data_te = train_inte.iloc[ind_te]
    d = data_tr.groupby('user_id')['click'].mean().to_dict()
    train_oov[ind_te] = data_te['user_id'].apply(lambda x: d.get(x, 0))

d = train_inte.groupby('user_id')['click'].mean().to_dict()
test_oov = test_inte['user_id'].apply(lambda x: d.get(x, 0))

cat_features = ['user_id']
num_features = ['time', 'duration_time']
train_inte['user_click_oof'] = train_oov
test_inte['user_click_oof'] = test_oov

num_features += ['user_click_oof']

dd = train_inte['user_id'].value_counts().to_dict()
train_inte['cat_count'] = train_inte['user_id'].apply(lambda x: dd.get(x, 0))
test_inte['cat_count'] = test_inte['user_id'].apply(lambda x: dd.get(x, 0))

num_features += ['cat_count']
scaler = MinMaxScaler()
enc = OneHotEncoder()
X_cat = enc.fit_transform(train_inte[cat_features])
X_num = scaler.fit_transform(train_inte[num_features])
X = ssp.hstack([X_cat,X_num]).tocsr()

X_t_cat = enc.transform(test_inte[cat_features])
X_t_num = scaler.transform(test_inte[num_features])
X_t = ssp.hstack([X_t_cat, X_t_num]).tocsr()
print X_t

y = train_inte['click'].values

del X_cat
del X_num
del X_t_cat
del X_t_num
gc.collect()

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(train_inte['user_id'], y)
# for ind_tr, ind_te in skf:
#     X_train = X[ind_tr]
#     X_test = X[ind_te]
#     y_train = y[ind_tr]
#     y_test = y[ind_te]
#     break
# del X
# gc.collect()

clf = LogisticRegression(C=10,random_state=0)
clf.fit(X, y)

y_sub = clf.predict_proba(X_t)[:, 1]

submission = pd.DataFrame()
submission['user_id'] = test_inte['user_id']
submission['photo_id'] = test_inte['photo_id']
submission['click_probability'] = y_sub
submission['click_probability'].apply(lambda x: float('%.6f' % x))
submission.to_csv('submission_lr20180516.csv', sep='\t', index=False)
