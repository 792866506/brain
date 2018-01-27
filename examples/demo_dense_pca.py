#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:39:32 2017

@author: al
"""

import torch
import mne
import numpy as np
from mne.io import concatenate_raws
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.signalproc import exponential_running_standardize

import  sys
sys.path.insert(0,'/home/al/braindecode/code/braindecode/braindecode')
from models.deep_dense import DeepDenseNet
from models.eeg_resnet import EEGResNet


import pickle
pkl_file = open('/home/al/braindecode/data/phys/train_set.pkl', 'rb')
train_set = pickle.load(pkl_file)#  (2125, 64, 497)
pkl_file.close()

pkl_file = open('/home/al/braindecode/data/phys/test_set.pkl', 'rb')
test_set = pickle.load(pkl_file)
pkl_file.close()


n_classes = 2
n_chans = train_set.X.shape[1]
input_time_length = train_set.X.shape[2]

from braindecode.models.deep4 import Deep4Net
from torch import nn
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model
from torch.nn.functional import elu
from torch.nn.functional import relu
# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = True
set_random_seeds(seed=20170629, cuda=cuda)

model = model = DeepDenseNet(in_chans= n_chans,
                     n_classes = n_classes,
                     input_time_length= input_time_length,
                     final_conv_length='auto',                               
                     bn_size=2,  ).create_network()

'''
model = EEGResNet( in_chans = in_chans,
                 n_classes = n_classes,
                 input_time_length = input_time_length,
                 final_pool_length = 'auto',
                 n_first_filters = 30,
                 n_layers_per_block=2,
                 first_filter_length=5,
                 nonlinearity=elu,
                 drop_prob = 0.5,
                 split_first_layer=True,
                 batch_norm_alpha=0.1,
                 batch_norm_epsilon=1e-4,).create_network()
'''
print model
if cuda:
    model.cuda()

from torch import optim

optimizer = optim.Adam(model.parameters())



from braindecode.torch_ext.util import np_to_var, var_to_np
import torch.nn.functional as F
from numpy.random import RandomState
import torch as th
from braindecode.datautil.iterators import get_balanced_batches
rng = RandomState((2017,6,30))


train_accu=[]
test_accu=[]

for i_epoch in range(50):
    i_trials_in_batch = get_balanced_batches(len(train_set.X), rng, shuffle=True,
                                           batch_size=50)
    # Set model to training mode
    model.train()
    for i_trials in i_trials_in_batch :
        batch_X = train_set.X[i_trials][:,:,:,None]
        batch_y = train_set.y[i_trials]
        net_in = np_to_var(batch_X)
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(batch_y)
        if cuda:
            net_target = net_target.cuda()
        # Remove gradients of last backward pass from all parameters
        optimizer.zero_grad()
        # Compute outputs of the network
        outputs = model(net_in)
        #outputs = th.mean(outputs, dim=2, keepdim=False)
        # Compute the loss
        loss = F.nll_loss(outputs, net_target)
        #loss = F.cross_entropy(outputs, net_target)
        # Do the backpropagation
        loss.backward()
        # Update parameters with the optimizer
        optimizer.step()

    # Print some statistics each epoch
    model.eval()
    print("Epoch {:d}".format(i_epoch))
#    for setname, dataset in (('Train', train_set), ('Test', test_set)):
    setname='Train'
    dataset=train_set              
    all_preds = []
    all_losses = []
    batch_sizes = []
    i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, shuffle=True,
                                        batch_size=50)
    for i_trials in i_trials_in_batch:
        batch_X = dataset.X[i_trials][:,:,:,None]
        batch_y = dataset.y[i_trials]
        net_in = np_to_var(batch_X)
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(batch_y)
        if cuda:
            net_target = net_target.cuda()
        outputs = model(net_in)
        #outputs = th.mean(outputs, dim=2, keepdim=False)
        loss = F.nll_loss(outputs, net_target)
        #loss = F.cross_entropy(outputs, net_target)
        predicted_labels = np.argmax(var_to_np(outputs), axis=1)
        accuracy = np.mean(batch_y  == predicted_labels)
        #print accuracy
        
        all_losses.append(var_to_np(loss))
        batch_sizes.append(len(batch_X))
        all_preds.append(accuracy)
    loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
                   np.mean(batch_sizes))
    print("{:6s} Loss: {:.5f}".format( setname, loss))
    
    
    accuracy = np.mean(np.array(all_preds) * np.array(batch_sizes) /
                   np.mean(batch_sizes))
    print("{:6s} Accuracy: {:.1f}%".format(
        setname, accuracy * 100))
    train_accu.append(accuracy)
    
    
    
    setname='Test'
    dataset=test_set              
    all_preds = []
    all_losses = []
    batch_sizes = []
    i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, shuffle=True,
                                        batch_size=50)
    for i_trials in i_trials_in_batch:
        batch_X = dataset.X[i_trials][:,:,:,None]
        batch_y = dataset.y[i_trials]
        net_in = np_to_var(batch_X)
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(batch_y)
        if cuda:
            net_target = net_target.cuda()
        outputs = model(net_in)
        #outputs = th.mean(outputs, dim=2, keepdim=False)
        loss = F.nll_loss(outputs, net_target)
        #loss = F.cross_entropy(outputs, net_target)
        predicted_labels = np.argmax(var_to_np(outputs), axis=1)
        accuracy = np.mean(batch_y  == predicted_labels)
        #print accuracy
        
        all_losses.append(var_to_np(loss))
        batch_sizes.append(len(batch_X))
        all_preds.append(accuracy)
    loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
                   np.mean(batch_sizes))
    print("{:6s} Loss: {:.5f}".format( setname, loss))
    
    accuracy = np.mean(np.array(all_preds) * np.array(batch_sizes) /
                   np.mean(batch_sizes))
    print("{:6s} Accuracy: {:.1f}%".format(
        setname, accuracy * 100))
    test_accu.append(accuracy)
    
    
'''
train_accu2=train_accu
test_accu2=test_accu 
'''
def plot_accu(train_accu=train_accu,test_accu=test_accu,a=80):
    import matplotlib.pyplot as plt
    plt.figure('accu')
    plt.xlabel('epoch')
    plt.ylabel('loss & accu')
    plt.yticks(np.linspace(0,1,num=30))
    plt.plot(range(a),train_accu[:a],'b.-',label='train',)
    plt.plot( range(a),test_accu[:a],'r.-',label='test')
    

    max_indx=np.argmax(test_accu)#max value index
    plt.plot(max_indx,test_accu[max_indx],'ks')
    show_max=str(max_indx)+' '+str(test_accu[max_indx])
    plt.annotate(show_max,xytext=(max_indx,test_accu[max_indx]),
                 xy=(max_indx,test_accu[max_indx]))
    plt.legend()
    plt.show()


   
def plot_accu2(train_accu,test_accu,
               train_accu2,test_accu2,a=80):
    import matplotlib.pyplot as plt
    plt.figure()#figsize=(8,8))
    plt.xlabel('epoch')
    plt.ylabel('loss & accu')
    plt.yticks(np.linspace(0,1,num=30))
    plt.plot(range(a),train_accu[:a],'b.-',label='train',)
    plt.plot( range(a),test_accu[:a],'r.-',label='test')
    plt.plot(range(a),train_accu2[:a],'.-',color='royalblue',label='train2',)
    plt.plot( range(a),test_accu2[:a],'.-',color='deeppink',label='test2')

    max_indx=np.argmax(test_accu)#max value index
    plt.plot(max_indx,test_accu[max_indx],'ks')
    show_max=str(max_indx)+'*'+str(test_accu[max_indx])
    plt.annotate(show_max,xytext=(max_indx,test_accu[max_indx]),
                 xy=(max_indx,test_accu[max_indx]))
    
    max_indx=np.argmax(test_accu2)#max value index
    plt.plot(max_indx,test_accu2[max_indx],'ks')
    show_max=str(max_indx)+'*'+str(test_accu2[max_indx])
    plt.annotate(show_max,xytext=(max_indx,test_accu2[max_indx]),
                 xy=(max_indx,test_accu2[max_indx]))
    
    plt.legend()
    plt.show()

        
