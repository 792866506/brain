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

subject_id=7
X = np.concatenate(
        [np.load('/home/al/braindecode/0_4500/conv_classifier_in{:1d}.npy'\
                .format(subject_id)).reshape(2,288,-1),
        np.load('/home/al/braindecode/fc_dense/conv_classifier_in{:1d}.npy'\
                .format(subject_id)).reshape(2,288,-1)],axis=2
                )
train = X[0]
test = X[1]

train_label = np.load('/home/al/braindecode/fc_dense/label{:1d}.npy'\
                .format(subject_id))[0]

test_label = np.load('/home/al/braindecode/fc_dense/label{:1d}.npy'\
                .format(subject_id))[1]

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
    return  alg

xgb1 = XGBClassifier(learning_rate =0.1,
                     n_estimators=1000,
                     max_depth=6,
                     min_child_weight=4,
                     gamma=0,
                     n_jobs = -1,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective= 'multi:softmax',
                     num_class=4,
                     silent =False,
                     seed=27)
modelfit(xgb1,cv_folds=5,early_stopping_rounds=100)


param_test1 = {
 'max_depth':[2,3,4],
 'min_child_weight':[1,2,3]
}

param_test4 = {
 'subsample':[i/100.0 for i in range(40,60,5)],
 'colsample_bytree':[i/100.0 for i in range(70,90,5)]
}



gsearch1 = GridSearchCV(
        estimator = XGBClassifier(learning_rate =0.1,
                                 n_estimators=34,                                  
                                 max_depth=3,
                                 min_child_weight=2,
                                 gamma=0, 
                                 n_jobs = -1,
                                 subsample=0.5, 
                                 colsample_bytree=0.8,
                                 objective= 'multi:softmax',
                                 scale_pos_weight=1, 
                                 num_class=4,
                                 seed=27), 
        param_grid = param_test1,  
         iid=False, 
         cv=5)
gsearch1.fit(train,train_label)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_




