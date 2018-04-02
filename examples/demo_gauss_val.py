#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 13:01:26 2018

@author: al
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 21:43:34 2018

@author: al
"""
import copy
import logging
import numpy as np
from collections import OrderedDict
import os
import torch.nn.functional as F
from torch import optim
import torch
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
                                                RuntimeMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.datautil.splitters import split_into_two_sets,concatenate_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.visualization.perturbation import  compute_stft_mul_inputs,\
                                        compute_stft_inputs
from numpy.random import RandomState

#from tensorboardX import SummaryWriter
import sys
#sys.path.insert(0,'/home/al/braindecode/code/braindecode/braindecode')
from braindecode.models.deep_dense import DeepDenseNet
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet


def get_dataset( sub, ival ,
                low_cut_hz = 0 ,# 0 or 4,
                high_cut_hz = 38):
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
    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)
    return train_set,test_set

    
def gaussian_perturbation(amps, rng):
    perturbation = rng.normal(0,0.002,amps.shape).astype(np.float32)
    return perturbation


if __name__ =='__main__':  
    log = logging.getLogger(__name__)
    cuda = True                  
    acc=[]
    path = 'shallow_002'
    for subject_id in range(1,2):
        ival = [-500, 4000]
        logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                            level=logging.DEBUG, stream=sys.stdout)
        
        train_set,test_set = get_dataset( subject_id, ival)
        X=copy.deepcopy(train_set.X)
        y=copy.deepcopy(train_set.y)
        train_set, valid_set = split_into_two_sets(train_set,first_set_fraction=0.8)
      
        for i in range(1):
            seed= i
            new_train_X_1=compute_stft_inputs(X, perturb_fn=gaussian_perturbation,
                                              framesz=0.1,hop=0.05,seed=seed)
            new_train_set_1 = SignalAndTarget(new_train_X_1,y)
            train_set  = concatenate_two_sets(new_train_set_1,train_set)
        print train_set.X.shape
        set_random_seeds(seed=20190706, cuda=cuda)
        
        n_classes = 4
        n_chans = 22
        input_time_length = 1125
        model = Deep4Net(in_chans= n_chans,
                     n_classes = n_classes,
                     input_time_length= input_time_length,
                     final_conv_length='auto',
                ).create_network()
        
        model.cuda()    
        log.info("Model: \n{:s}".format(str(model)))
        optimizer = optim.Adam(model.parameters())
        iterator = BalancedBatchSizeIterator(batch_size=64)
        
        stop_criterion = Or([MaxEpochs(1000),
                             NoDecrease('valid_misclass', 80)])
        monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]
        model_constraint = MaxNormDefaultConstraint()
    
        exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                         loss_function=F.nll_loss, optimizer=optimizer,
                         model_constraint=model_constraint,
                         monitors=monitors,
                         stop_criterion=stop_criterion,
                         remember_best_column='valid_misclass',
                         run_after_early_stop=True, cuda=cuda)
        exp.run()

        
        log.info("Last 10 epochs")
        log.info("\n" + str(exp.epochs_df.iloc[-10:]))
        log.info("\n" + str(1-exp.epochs_df['test_misclass'].iloc[-10:]))
        acc.append(1-exp.epochs_df['test_misclass'].iloc[-1])
        
        
        torch.save(exp.model,'/home/al/braindecode/result/'+path+'/sub{:1d}'\
                   .format(subject_id))
        
print np.mean(acc)