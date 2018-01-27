#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:55:51 2018

@author: al
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

subject_id=2
train = np.load('/home/al/braindecode/code/braindecode/xg/train_conv_in{:1d}.npy'
                .format(subject_id))
train_label = np.load('/home/al/braindecode/code/braindecode/xg/train_label{:1d}.npy'
                .format(subject_id))
test = np.load('/home/al/braindecode/code/braindecode/xg/test_conv_in{:1d}.npy'
                .format(subject_id))
test_label = np.load('/home/al/braindecode/code/braindecode/xg/test_label{:1d}.npy'
                .format(subject_id))

def modelfit(alg,useTrainCV=True, cv_folds=5, early_stopping_rounds=60):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train,train_label)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
               early_stopping_rounds=early_stopping_rounds, show_stdv=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(train, train_label)
    
    #Predict training set:
    dtrain_predictions = alg.predict(train)
    dtest_predictions = alg.predict(test)

    #Print model report:
    print "\nModel Report"
    print  sum(int(dtrain_predictions[i]) == train_label[i] \
                    for i in range(len(train_label))) / float(len(train_label))
    print  sum(int(dtest_predictions[i]) == test_label[i] \
                    for i in range(len(test_label))) / float(len(test_label))
    

xgb1 = XGBClassifier(learning_rate =0.1,
                     n_estimators=500,
                     max_depth=5,
                     min_child_weight=3,
                     gamma=0,
                     subsample=0.9,
                     colsample_bytree=0.8,
                     objective= 'multi:softmax',
                     num_class=4,
                     seed=27)
modelfit(xgb1,cv_folds=5)



train_X = train
train_Y = train_label

test_X = test
test_Y = test_label
xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)


# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['min_child_weight'] = 3
param['silent'] = 1
param['num_class'] = 4
param['subsample']=0.9
param['colsample_bytree']=0.8
param['seed']=100

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 300
bst = xgb.train(param, xg_train, num_round, watchlist,
                early_stopping_rounds=60)
# get prediction
pred = bst.predict(xg_test)
error_rate = np.sum(pred == test_Y) / float(test_Y.shape[0])
print('Test accu using softmax = {}'.format(error_rate))

