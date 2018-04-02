#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:19:05 2018

@author: al
"""

 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import  numpy as np
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)

def save_file(subject_id = 1):
    for event_id in xrange(1,5):
        file_name = '/home/al/braindecode/data/P300/sub{:01d}/event{:01d}.txt'.\
                            format(subject_id,event_id)
        aa=np.loadtxt(file_name)
        np.save('/home/al/braindecode/data/P300/sub{:01d}/event{:01d}.npy'.\
                            format(subject_id,event_id),aa)


from sklearn import preprocessing

def load_p300(subject_id, n_classes, rng, shuffle):

    X_list=[]
    y_list=[]
    for event_id in xrange(1,5):
        file_name = '/home/al/braindecode/data/P300/sub{:01d}/event{:01d}.npy'.\
                            format(subject_id,event_id)
        aa=np.load(file_name)[:60]
        preprocessing.scale(aa,axis=1,copy=False)
        aa=aa.reshape(60,-1,1000)
        X_list.append(aa.transpose(1,0,2))
        label = np.array([event_id]*(aa.shape[1]))
        y_list.append(label)
    X=np.concatenate(X_list,axis=0)
    y=np.concatenate(y_list,axis=0)

#    min_array=X.reshape(400,-1).min(axis=1)
#    max_array=X.reshape(400,-1).max(axis=1)
#    filt_index = np.where((max_array<35) & (min_array>-35))[0]
#    X = X[filt_index]
#    print X.shape
#    y = y[filt_index]
#    X=[exponential_running_standardize(a.T, factor_new=1e-3, \
#                                     init_block_size=900, eps=1e-4).T for a in X]
#    X= np.array(X)
    X = (X[:,:,:,np.newaxis]).astype('float32')
    y = (y-1).astype('int64')

    if n_classes==2:
        y=np.where(y==2, 0, y)
        y=np.where(y==3 ,1, y)
    
    all_inds = np.array(range(X.shape[0]))
    if shuffle==True:
        rng.shuffle(all_inds)
    return X[all_inds],y[all_inds]





