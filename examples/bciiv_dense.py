#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:45:38 2017

@author: al
"""

import logging
import os.path
from collections import OrderedDict

import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.nn.functional import elu
from torch.nn.functional import relu

from braindecode.models.deep4 import Deep4Net
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var,var_to_np
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
import  sys
sys.path.insert(0,'/home/al/braindecode/code/braindecode/braindecode')
from models.eeg_densenet import EEGDenseNet
from models.eeg_resnet import EEGResNet
from models.deep_dense import DeepDenseNet
log = logging.getLogger(__name__)


logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
    # Should contain both .gdf files and .mat-labelfiles from competition
data_folder = '/home/al/BCICIV_2a_gdf/'
subject_id = 8 # 1-9
low_cut_hz = 0 # 0 or 4
model = 'deep_dense' #'shallow' or 'deep'
cuda = True
exp_std =True
pca=False


train_filename = 'A{:02d}T.gdf'.format(subject_id)
test_filename = 'A{:02d}E.gdf'.format(subject_id)
train_filepath = os.path.join(data_folder, train_filename)
test_filepath = os.path.join(data_folder, test_filename)
train_label_filepath = train_filepath.replace('.gdf', '.mat')
test_label_filepath = test_filepath.replace('.gdf', '.mat')

train_loader = BCICompetition4Set2A(
    train_filepath, labels_filename=train_label_filepath)
test_loader = BCICompetition4Set2A(
    test_filepath, labels_filename=test_label_filepath)
train_cnt = train_loader.load()
test_cnt = test_loader.load()

# Preprocessing

train_cnt = train_cnt.drop_channels(['STI 014', 'EOG-left',
                                     'EOG-central', 'EOG-right'])
assert len(train_cnt.ch_names) == 22
# lets convert to millvolt for numerical stability of next operations
train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
train_cnt = mne_apply(
    lambda a: bandpass_cnt(a, low_cut_hz, 38, train_cnt.info['sfreq'],
                           filt_order=3,
                           axis=1), train_cnt)
if exp_std:
    train_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        train_cnt)

if pca:
    from sklearn.decomposition import  PCA
    pca=PCA(n_components=len(train_cnt.ch_names),copy=True) 
    train_cnt = mne_apply(
        lambda a: pca.fit_transform(a.T).T,
        train_cnt)
    
test_cnt = test_cnt.drop_channels(['STI 014', 'EOG-left',
                                   'EOG-central', 'EOG-right'])
assert len(test_cnt.ch_names) == 22
test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
test_cnt = mne_apply(
    lambda a: bandpass_cnt(a, low_cut_hz, 38, test_cnt.info['sfreq'],
                           filt_order=3,
                           axis=1), test_cnt)
if exp_std:
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        test_cnt)
        
if pca:
    from sklearn.decomposition import  PCA
    pca=PCA(n_components=len(train_cnt.ch_names),copy=True) 
    test_cnt = mne_apply(
        lambda a: pca.fit_transform(a.T).T,
        test_cnt)      

marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                         ('Foot', [3]), ('Tongue', [4])])
ival = [-500, 4000]

train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)


set_random_seeds(seed=20190706, cuda=cuda)

n_classes = len(marker_def)
n_chans = int(train_set.X.shape[1])
input_time_length = train_set.X.shape[2]
if model == 'shallow':
    model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                        final_conv_length='auto').create_network()
elif model == 'deep':
    model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                        final_conv_length='auto').create_network()
elif model == 'dense':
    model = EEGDenseNet(in_chans = n_chans,
             n_classes = n_classes,
             input_time_length = input_time_length,
             final_pool_length= 'auto',
             first_filter_length=11,
             nonlinearity=elu,
             split_first_layer=True,
             batch_norm_alpha=0.1,
             growth_rate=10, 
             bn_size=4, 
             drop_rate=0.5, 
             block_config=(2,2,2,2,2),#2 2 2 2 2 2   50
             compression=0.5,
             num_init_features=25, 
             ).create_network()
    
    
elif model == 'deep_dense':
    model = DeepDenseNet(in_chans= n_chans,
                 n_classes = n_classes,
                 input_time_length= input_time_length,
                  final_conv_length='auto',
                 bn_size=2, 
                 ).create_network()
if cuda:
    model.cuda()
log.info("Model: \n{:s}".format(str(model)))

optimizer = optim.Adam(model.parameters())


from numpy.random import RandomState
from braindecode.datautil.iterators import get_balanced_batches
rng = RandomState((2017,6,30))


train_accu=[]
test_accu=[]

for i_epoch in range(200):
    i_trials_in_batch = get_balanced_batches(len(train_set.X), rng, shuffle=True,
                                           batch_size=60)
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
                                        batch_size=200)
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
                                        batch_size=200)
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
def plot_accu(train_accu=train_accu,test_accu=test_accu,a=90):
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

    

