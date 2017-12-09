#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:20:52 2017

@author: al
"""

import logging
import os.path
from collections import OrderedDict

import numpy as np
import torch.nn.functional as F
from torch import optim
import torch as th
from torch.nn.functional import elu
from torch.nn.functional import relu

from braindecode.models.deep4 import Deep4Net
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.torch_ext.util import set_random_seeds, np_to_var ,var_to_np
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

from tensorboardX import SummaryWriter

import  sys
sys.path.insert(0,'/home/al/braindecode/code/braindecode/braindecode')
from models.eeg_densenet import EEGDenseNet
from models.eeg_resnet import EEGResNet
from models.deep_dense import DeepDenseNet
log = logging.getLogger(__name__)
log_dir=None#'runs/model1' #None
comment = 'dense'
n_epoch = 150
model = 'deep' #'shallow' or 'deep'
cuda = True
exp_stand = True
subject_id = 3 # 1-9
low_cut_hz = 0 # 0 or 4
high_cut_hz = 38
data_folder = '/home/al/BCICIV_2a_gdf/'
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
    lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, train_cnt.info['sfreq'],
                           filt_order=3,
                           axis=1), train_cnt)

if  exp_stand==True:
    train_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        train_cnt)

    
test_cnt = test_cnt.drop_channels(['STI 014', 'EOG-left',
                                   'EOG-central', 'EOG-right'])
assert len(test_cnt.ch_names) == 22
test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
test_cnt = mne_apply(
    lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt.info['sfreq'],
                           filt_order=3,
                           axis=1), test_cnt)

if  exp_stand==True:
    
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        test_cnt)


marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                          ('Foot', [3]), ('Tongue', [4])])
'''
marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],)])
'''
ival = [-500, 4000]

train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)



set_random_seeds(seed=20190706, cuda=cuda)

n_classes = len(marker_def)
n_chans = int(train_set.X.shape[1])
input_time_length=1000

if model == 'shallow':
    model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length=30).create_network()
    to_dense_prediction_model(model)
elif model == 'deep':
    model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length=2).create_network()
    to_dense_prediction_model(model)
elif model == 'dense':
    model = EEGDenseNet(in_chans = n_chans,
             n_classes = n_classes,
             input_time_length = input_time_length,
             final_pool_length= 2,
             first_filter_length=11,
             nonlinearity=elu,
             split_first_layer=True,
             batch_norm_alpha=0.1,
             growth_rate=20, 
             bn_size=4, 
             drop_rate=0.5, 
             block_config=(3,2),#2 2 2 2 2 2   50
             compression=0.5,
             num_init_features=20, 
             ).create_network()
    
elif model == 'deep_dense':
    model = DeepDenseNet(in_chans= n_chans,
                 n_classes = n_classes,
                 input_time_length= input_time_length,
                  final_conv_length=2,
                 bn_size=2, 
                 ).create_network()
    #to_dense_prediction_model(model)


def _set_lr(optimizer, epoch, n_epochs, lr):
    lr = lr
    if float(epoch) / n_epochs > 0.75:
        lr = lr * 0.01
    elif float(epoch) / n_epochs > 0.5:
        lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(param_group['lr'])

if cuda:
    model.cuda()

log.info("Model: \n{:s}".format(str(model)))
dummy_input = np_to_var(train_set.X[:1, :, :, None])
if cuda:
    dummy_input = dummy_input.cuda()
out = model(dummy_input)

n_preds_per_input = out.size()[2]
print("{:d} predictions per input/trial".format(n_preds_per_input))


optimizer = optim.Adam(model.parameters())
iterator = CropsFromTrialsIterator(batch_size=64,
                                   input_time_length=input_time_length,
                                   n_preds_per_input=n_preds_per_input)


from numpy.random import RandomState
from braindecode.datautil.iterators import get_balanced_batches
rng = RandomState((2017,6,30))
train_accu=[]
test_accu=[]


#########################################################
writer = SummaryWriter(log_dir,comment=comment)
for i_epoch in range(n_epoch):#origin 20
    print("Epoch {:d}".format(i_epoch))
    print("Train....")
    # Set model to training mode
    model.train()
    for batch_X, batch_y in iterator.get_batches(train_set, shuffle=False):
        net_in = np_to_var(batch_X)
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(batch_y)
        if cuda:
            net_target = net_target.cuda()
        # Remove gradients of last backward pass from all parameters
        optimizer.zero_grad()
        outputs = model(net_in)
        # Mean predictions across trial
        # Note that this will give identical gradients to computing
        # a per-prediction loss (at least for the combination of log softmax activation
        # and negative log likelihood loss which we are using here)
        outputs = th.mean(outputs, dim=2, keepdim=False)
        loss = F.nll_loss(outputs, net_target)
        loss.backward()
        optimizer.step()

    model.eval()
#    for setname, dataset in (('Train', train_set), ('Test', test_set)):
    setname='Train'
    dataset=train_set              
    all_preds = []
    all_losses = []
    batch_sizes = []
    i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, shuffle=True,
                                        batch_size=100)
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
        outputs = th.mean(outputs, dim=2, keepdim=False)
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
    
    writer.add_scalars('data/loss', {"train_loss":loss,
                                      }, i_epoch)
    writer.add_scalars('data/accu', {  "train_accu":accuracy,
                                     }, i_epoch)
    
    setname='Test'
    dataset=test_set              
    all_preds = []
    all_losses = []
    batch_sizes = []
    i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, shuffle=True,
                                        batch_size=100)
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
        outputs = th.mean(outputs, dim=2, keepdim=False)
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
    ###################################
    writer.add_scalars('data/loss', {"test_loss":loss,
                                      }, i_epoch)
    writer.add_scalars('data/accu', {  "test_accu":accuracy,
                                     }, i_epoch)
########################
writer.add_graph(model, outputs.cpu())   
#writer.export_scalars_to_json("./all_scalars.json")
writer.save_txt(str(model),'model1.txt')
writer.close()
