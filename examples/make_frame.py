#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:13:17 2017

@author: al
"""


'''
run  bciiv_2a  first 
test the model  on 'deep_dense'

result_array  is 
Out[38]: 
array([[69,  2,  1,  0],
       [ 4, 53, 10,  5],
       [ 1,  5, 58,  8],
       [ 1,  1,  6, 64]])

'''
import numpy as np
from collections import OrderedDict

labels =exp.all_targets
preds  = exp.all_preds



left_index =  np.where(labels == 0)[0]
right_index = np.where(labels == 1)[0]
foot_index =  np.where(labels == 2)[0]
tongue_index = np.where(labels == 3)[0]

index_dict=[('left_index',left_index),
                         ('right_index',right_index),
                         ('foot_index',foot_index),
                         ('tongue_index',tongue_index)]

result_list=[]
for name,index in index_dict:
    a=np.sum(preds[index]==0)
    b=np.sum(preds[index]==1)
    c=np.sum(preds[index]==2)
    d=np.sum(preds[index]==3)
    result_list.append(np.array([a,b,c,d]))
result_array = np.array(result_list)




