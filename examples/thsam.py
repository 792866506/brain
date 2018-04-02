#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:26:08 2018

@author: al
"""

import os.path
from collections import OrderedDict


import numpy as np
import torch
import torch.nn.functional as F

from braindecode.models.deep4 import Deep4Net
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                          exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne



#from tensorboardX import SummaryWriter
import sys
sys.path.insert(0,'/home/al/braindecode/code/braindecode/braindecode')
from models.deep_dense import DeepDenseNet


from torchsample.modules import ModuleTrainer
from torchsample.callbacks import EarlyStopping, ReduceLROnPlateau
from torchsample.regularizers import L1Regularizer, L2Regularizer
from torchsample.constraints import UnitNorm,MaxNorm
from torchsample.initializers import XavierUniform
from torchsample.metrics import CategoricalAccuracy
from torchsample import TensorDataset

import os
from torch.utils.data import DataLoader

def run_exp(data_folder, subject_id, low_cut_hz,high_cut_hz, model_name):  
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
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        test_cnt)
    
      
        
    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])
    ival = [-500, 4000]
    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)
               
    train_set, valid_set = split_into_two_sets(train_set,
                                               first_set_fraction=0.8)


    train_dataset = TensorDataset(train_set.X[:,:,:,np.newaxis], train_set.y)
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_dataset = TensorDataset(valid_set.X[:,:,:,np.newaxis], valid_set.y)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_dataset = TensorDataset(test_set.X[:,:,:,np.newaxis], test_set.y)
    test_loader = DataLoader(val_dataset, batch_size=32)

    
    n_classes = 4
    n_chans = int(train_loader.dataset.inputs[0].shape[1])
    input_time_length = int(train_loader.dataset.inputs[0].shape[2])
    if model_name == 'shallow':
        model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length='auto').create_network()
    elif model_name == 'deep':
        model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length='auto').create_network()
        
    elif model_name== 'deep_dense':
        model = DeepDenseNet(in_chans= n_chans,
                     n_classes = n_classes,
                     input_time_length= input_time_length,
                     final_conv_length= 8 ,                               
                     bn_size=2,  ).create_network()

    trainer = ModuleTrainer(model)
    
    callbacks = [EarlyStopping(patience=100),
                 ReduceLROnPlateau(factor=0.5, patience=60)]
    regularizers = [L1Regularizer(scale=1e-3, module_filter='conv*'),
                    L2Regularizer(scale=1e-5, module_filter='fc*')]
    constraints = [MaxNorm(value=2., frequency=5, unit='batch', module_filter='*fc*')]
    initializers = [XavierUniform(bias=False, module_filter='fc*')]
    
    trainer.compile(loss='nll_loss',
                    optimizer='adam',
                    regularizers=regularizers,
                    constraints=constraints,
                    initializers=initializers,
                    callbacks=callbacks)

    summary = trainer.summary([22,1125,1])
    print(summary)
    
    trainer.fit_loader(train_loader, val_loader, num_epoch=1000, verbose=1)
    return trainer
    
    
if __name__ == '__main__':
    mean=[]
    mini=[]
    confuse_mat =[]
    train = True                    
    path='fc_dense_relu'
    for subject_id in xrange(1,2):
        data_folder = '/home/al/BCICIV_2a_gdf/'
        low_cut_hz = 0 # 0 or 4
        high_cut_hz = 38
        model_name = 'deep_dense' #'shallow' or 'deep'  cnn++  Dense_LSTM  RCNN_EEG  resnet
        cuda = True
        if train:
            exp = run_exp(data_folder, subject_id, low_cut_hz,high_cut_hz, model_name)                     
            #torch.save(exp.model,'/home/al/braindecode/'+path+'/bci_{:01d}_model.pkl'.format(subject_id))

