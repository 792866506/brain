#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:19:36 2018

@author: al
"""

import numpy as np
path_list=['fc_dense','-300_4200']
subject_id = 1

train_conv_in=[]
test_conv_in=[]

train_soft_in=[]
test_soft_in=[]

train_y=[]
test_y=[]
for subject_id in xrange(1,10):
        train_conv_in.append([np.load('/home/al/braindecode/'+path+'/conv_classifier_in{:1d}.npy'\
                                     .format(subject_id))[0] for  path in path_list])
        test_conv_in.append([np.load('/home/al/braindecode/'+path+'/conv_classifier_in{:1d}.npy'\
                                     .format(subject_id))[1] for  path in path_list])
            
        train_soft_in.append([ np.load('/home/al/braindecode/'+path+'/softmax_in{:1d}.npy'.format
                    (subject_id))[0]  for  path in path_list ])
        test_soft_in.append([ np.load('/home/al/braindecode/'+path+'/softmax_in{:1d}.npy'.format
                    (subject_id))[1]  for  path in path_list ])
        
        train_y.append( [np.load('/home/al/braindecode/'+path+'/label{:1d}.npy'.format\
                    (subject_id))[0]  for  path in path_list ])
        test_y.append( [np.load('/home/al/braindecode/'+path+'/label{:1d}.npy'.format\
                    (subject_id))[1]  for  path in path_list ])


true_label = np.array(label).reshape(-1)
softmax_in_array=np.concatenate(softmax_in,axis=0)[:,:,0,0]
softmax_in_array_1=np.concatenate(softmax_in_1,axis=0)[:,:,0,0]
softmax_out_array=np.concatenate(softmax_out,axis=0)[:,:,0,0]
softmax_out_array_1=np.concatenate(softmax_out_1,axis=0)[:,:,0,0]

m = nn.Softmax()
prob=var_to_np(m(np_to_var(softmax_in_array)))
prob_1=var_to_np(m(np_to_var(softmax_in_array_1)))


pred_label=np.argmax(0.5*prob+0.5*prob_1,axis=1)
#pred_label = np.argmax(np.maximum(prob, prob_1),axis=1)
print np.sum(pred_label==true_label)/288.0