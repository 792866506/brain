# -*- coding: utf-8 -*-
'''
get three mat file

'''
import logging
import os.path

from collections import OrderedDict


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.functional import elu
from torch.nn.functional import relu

from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.util import set_random_seeds
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

log = logging.getLogger(__name__)

from numpy.random import RandomState
rng = RandomState((2017,6,30))
import sys
sys.path.insert(0,'/home/al/braindecode/code/braindecode/braindecode')
from models.eeg_densenet import EEGDenseNet
from models.eeg_resnet import EEGResNet
from models.deep_dense import DeepDenseNet
from models.CNNPlus import CNNPlus
from models.Dense_LSTM import Dense_LSTM
from models.RCNN_EEG import RCNN_EEG
from models.resnet  import EEGResNet
from braindecode.datautil.signal_target import  SignalAndTarget

acc_list=[]
softmax_in=[]
softmax_out=[]
conv_classifier_in=[]
label=[]
def crop(train_set):
    X=train_set.X.copy()
    aa=X[:,:,:1024]
    bb=X[:,:,101:]
    X=np.concatenate([aa,bb],axis=0)
    label_aa = train_set.y.copy()
    label_bb = train_set.y.copy()
    y=np.concatenate([label_aa,label_bb],axis=0)
    return SignalAndTarget(X,y)
def softmax_hook(module, inpt, output):
    print(module)
    softmax_in.append(var_to_np(inpt[0]))
    softmax_out.append (var_to_np(output))
    print("input  val:",inpt[0].size())
    print("output val:", output.size())
def conv_classifier_hook(module, inpt, output):
    print(module)
    conv_classifier_in.append(var_to_np(inpt[0]))
    print("input  val:",inpt[0].size())
    print("output val:", output.size())


path='shallow_aug'
ival = [-500, 4000]
for subject_id in xrange(1,10):
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
    # Should contain both .gdf files and .mat-labelfiles from competition
    data_folder = '/home/al/BCICIV_2a_gdf/'
    low_cut_hz = 0 # 0 or 4
    cuda = True   
    model_name = '/home/al/braindecode/result/'+path+'/sub{:01d}'.format(subject_id)
    #deep_dense_model   shallow_model  se_dense  se_32_16  dense_enter_se
    
    
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
        lambda a: bandpass_cnt(a, low_cut_hz, 38, test_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt)
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        test_cnt)

    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])
    

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)
    set_random_seeds(seed=20190706, cuda=cuda)

    
    model = torch.load(model_name)
    handle_softmax = model.softmax.register_forward_hook(softmax_hook)
    handle_conv = model.conv_classifier.register_forward_hook(conv_classifier_hook)
        
    
    if cuda:
        model.cuda()    
    log.info("Model: \n{:s}".format(str(model)))

    optimizer = optim.Adam(model.parameters())
    
    setname = 'train'
    dataset=test_set 
    
    all_preds = []
    all_losses = []
    batch_sizes = []
    i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, shuffle=False,
                                        batch_size=len(dataset.X))
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
        loss = F.nll_loss(outputs, net_target)
        predicted_labels = np.argmax(var_to_np(outputs), axis=1)
        accuracy = np.mean(batch_y  == predicted_labels)        
        all_losses.append(var_to_np(loss))
        batch_sizes.append(len(batch_X))
        all_preds.append(accuracy)
    loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
                   np.mean(batch_sizes))
    print("{:6s} Loss: {:.5f}".format(setname, loss))
    
    accuracy = np.mean(np.array(all_preds) * np.array(batch_sizes) /
                   np.mean(batch_sizes))
    print("{:6s} Accuracy: {:.4f}%".format(
        setname, accuracy * 100))
    acc_list.append(accuracy)
    


    import scipy.io as sio
    prob = softmax_in[0].reshape(288,4)################
    m = nn.Softmax(dim=-1)
    prob=var_to_np(m(np_to_var(prob)))
    sio.savemat('/home/al/braindecode/result/'+path+'/decision_values{:1d}.mat'
                .format(subject_id),{'decision_values':prob})
    sio.savemat('/home/al/braindecode/result/'+path+'/actual_label{:1d}.mat'
                .format(subject_id),{'actual_label':batch_y+1})
    sio.savemat('/home/al/braindecode/result/'+path+'/predict_label{:1d}.mat'
                .format(subject_id),{'predict_label':predicted_labels+1})

    