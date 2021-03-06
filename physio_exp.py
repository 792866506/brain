#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:58:29 2018

@author: al
"""

import logging
import os.path
import mne
import numpy as np
from mne.io import concatenate_raws
from braindecode.datautil.signal_target import SignalAndTarget
from collections import OrderedDict
import sys
sys.path.insert(0,'/home/al/braindecode/code/braindecode')
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.functional import elu
from torch.nn.functional import relu
#from braindecode.models.deep4 import Deep4Net
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
#from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.datautil.signal_target import SignalAndTarget
log = logging.getLogger(__name__)

from tensorboardX import SummaryWriter

sys.path.insert(0,'/home/al/braindecode/code/braindecode/braindecode')
from models.eeg_densenet import EEGDenseNet
from models.eeg_resnet import EEGResNet
from models.deep_dense import DeepDenseNet
from models.CNNPlus import CNNPlus
from models.Dense_LSTM import Dense_LSTM
from models.RCNN_EEG import RCNN_EEG
from models.resnet  import EEGResNet
from models.shallow_fbcsp  import ShallowFBCSPNet

def for_hook(module, inpt, output):
    print(module)
    print("input val:",inpt[0].size())
    print("output val:", output.size())

def run_exp(data_num,data_folder, subject_id, low_cut_hz, model_name, cuda,pca=False):
    physionet_paths = [ mne.datasets.eegbci.load_data(sub_id,[4,8,12,]) for sub_id in range(1,31)]#(1,51)
    physionet_paths = np.concatenate(physionet_paths)
    parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto')
             for path in physionet_paths]
    
    raw = concatenate_raws(parts)
    #raw = raw.drop_channels(  )
    
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    
    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')
    
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epoched = mne.Epochs(raw, events, dict(left=2, right=3,), tmin=1, tmax=4.1, proj=False, picks=picks,
                    baseline=None, preload=True)
    
    train_X = (epoched.get_data() * 1e6).astype(np.float32)
    train_y = (epoched.events[:,2] - 2).astype(np.int64) #2,3 -> 0,1
    
    physionet_paths_test = [mne.datasets.eegbci.load_data(sub_id,[4,8,12,]) for sub_id in range(31,36)]#51,56
    physionet_paths_test = np.concatenate(physionet_paths_test)
    parts_test = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto')
             for path in physionet_paths_test]
    raw_test = concatenate_raws(parts_test)
    
    picks_test = mne.pick_types(raw_test.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    
    events_test = mne.find_events(raw_test, shortest_event=0, stim_channel='STI 014')
    
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epoched_test = mne.Epochs(raw_test, events_test, dict(left=2, right=3), tmin=1, tmax=4.1, proj=False, picks=picks_test,
                    baseline=None, preload=True)
    
    test_X = (epoched_test.get_data() * 1e6).astype(np.float32)
    test_y = (epoched_test.events[:,2] - 2).astype(np.int64) #2,3 -> 0,1
    
    
    train_set = SignalAndTarget(train_X, y=train_y)
    test_set = SignalAndTarget(test_X, y=test_y)
    
    train_set, valid_set = split_into_two_sets(train_set,
                                               first_set_fraction=0.75)
    
    
    #  data augument
    mix_num = data_num
    new_data=[]
    new_label=[]
    for mix_label in [0,1]:
        index=np.where(train_set.y==mix_label)[0]
        for ite in xrange(mix_num):
            np.random.shuffle(index)
            for i in xrange(10):
                new_data.append(
                        np.sum([train_set.X[i],train_set.X[i+mix_num]],axis=0)/2.0
                        )
        new_label.extend([mix_label]*10*mix_num)
        
        
    new_label = np.array(new_label)
    new_train_set=SignalAndTarget(np.concatenate([train_set.X,np.array(new_data)],axis=0),
                    np.concatenate([train_set.y,new_label],axis=0))
    
    set_random_seeds(seed=20190706, cuda=cuda)
    n_classes = 2
    n_chans = int(train_set.X.shape[1])
    input_time_length = train_set.X.shape[2]
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
                     final_conv_length='auto',                               
                     bn_size=2,  ).create_network()
    elif model_name == 'cnn++':
        model = CNNPlus(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length='auto').create_network()
    elif model_name == 'Dense_LSTM':
        model = Dense_LSTM(n_chans, n_classes, input_time_length=input_time_length)
    elif model_name == 'RCNN_EEG':
        model = RCNN_EEG(n_chans, n_classes, input_time_length=input_time_length)
    elif model_name == 'resnet':
        model = EEGResNet(n_chans, n_classes, input_time_length=input_time_length,
                          final_pool_length='auto', n_first_filters=48).create_network()
            
    if cuda:
        model.cuda()    
    log.info("Model: \n{:s}".format(str(model)))

    optimizer = optim.Adam(model.parameters())

    iterator = BalancedBatchSizeIterator(batch_size=60)

    stop_criterion = Or([MaxEpochs(1000),
                         NoDecrease('valid_misclass', 150)])
    monitors = [LossMonitor(),MisclassMonitor()]
    model_constraint = MaxNormDefaultConstraint()
    '''
    exp = Experiment(model, new_train_set, valid_set, test_set,
                     iterator=iterator,
                     loss_function=F.nll_loss, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=False, cuda=cuda)
    exp.run()
    
    model = exp.model
    if cuda:
        model.cuda()            
    stop_criterion = Or([MaxEpochs(100),
                         NoDecrease('valid_misclass', 100)])
    exp = Experiment(model, train_set, valid_set, test_set,
                     iterator=iterator,
                     loss_function=F.nll_loss, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=False, cuda=cuda)
    exp.new_run()# not  remember  model  
    
    
    model = exp.model
    if cuda:
        model.cuda()            
    stop_criterion = Or([MaxEpochs(1000),
                         NoDecrease('valid_misclass', 100)])
    monitors = [LossMonitor(),MisclassMonitor()]
    '''
    exp = Experiment(model, train_set, valid_set, test_set,
                     iterator=iterator,
                     loss_function=F.nll_loss, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=True, cuda=cuda)
    exp.run()
    
    return exp

def test_exp(data_folder, subject_id, low_cut_hz, model_name, cuda,
             class_label=None,pca=False):
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
    ival = [-500, 4000]

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)
          
    train_set, valid_set = split_into_two_sets(train_set,
                                               first_set_fraction=0.8)

    set_random_seeds(seed=20190706, cuda=cuda)

    
    model = torch.load(model_name)
    #handle = model.softmax.register_forward_hook(for_hook)
        
    
    if cuda:
        model.cuda()    
    log.info("Model: \n{:s}".format(str(model)))

    optimizer = optim.Adam(model.parameters())

    iterator = BalancedBatchSizeIterator(batch_size=288)

    stop_criterion = Or([MaxEpochs(1),
                         NoDecrease('valid_misclass', 1)])
    monitors = [MisclassMonitor()]

    model_constraint = MaxNormDefaultConstraint()
    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=F.nll_loss, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=True, cuda=cuda)
   
    exp.run_eval()
    return exp

'''
if __name__ == '__main__':    
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
    # Should contain both .gdf files and .mat-labelfiles from competition
    data_folder = '/home/al/BCICIV_2a_gdf/'
    subject_id = 8 # 1-9
    low_cut_hz = 0 # 0 or 4
    model = 'deep_dense' #'shallow' or 'deep'  CNNPlus
    cuda = True
    exp = run_exp(data_folder, subject_id, low_cut_hz, model, cuda)
    log.info("Last 10 epochs")
    log.info("\n" + str(exp.epochs_df.iloc[-10:]))
    np.mean(exp.epochs_df.iloc[-10:]['test_misclass'])
    np.min(exp.epochs_df.iloc[-10:]['test_misclass'])    
    
'''
if __name__ == '__main__':
    mean=[]
    mini=[]
    train = True
    confuse_mat =[]
    data_num = 1
    for subject_id in xrange(1,2):
        logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                            level=logging.DEBUG, stream=sys.stdout)
        # Should contain both .gdf files and .mat-labelfiles from competition
        data_folder = '/home/al/BCICIV_2a_gdf/'
        low_cut_hz = 0 # 0 or 4
        model_name = 'deep_dense' #'shallow' or 'deep'  cnn++  Dense_LSTM  RCNN_EEG  resnet
        cuda = True
        if train:
            exp = run_exp(data_num,data_folder, subject_id, low_cut_hz, model_name, cuda)
            log.info("Last 10 epochs")
            log.info("\n" + str(exp.epochs_df.iloc[-10:]))
            log.info("\n" + str(1-exp.epochs_df['test_misclass'].iloc[-10:]))
            mean.append( np.mean(exp.epochs_df.iloc[-10:]['test_misclass']))
            mini.append( np.min(exp.epochs_df.iloc[-10:]['test_misclass']))
            file_name='/home/al/braindecode/data_aug/num{:02d}_bci_{:01d}_model.pkl'.format(data_num,subject_id)        
            torch.save(exp.model,file_name)
        else :
            model_name = '/home/al/braindecode/data_aug/num{:02d}_bci_{:01d}_model.pkl'.format(data_num,subject_id)
            #deep_dense_model   shallow_model  se_dense  se_32_16  dense_enter_se
            exp = test_exp(data_folder, subject_id, low_cut_hz, model_name, cuda)
            log.info("\n" + str(exp.epochs_df))
            print (1-exp.epochs_df)*100
            
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
            confuse_mat.append(result_array)
    new_confuse_mat = np.array(confuse_mat)
'''
mean=[]  
for aa in confuse_mat:
    mean.append(np.sum(aa[(0,1,2,3),(0,1,2,3)])/288.0)
np.mean(mean)

'''
