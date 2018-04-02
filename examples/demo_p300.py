#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:37:10 2018

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


import sys

from braindecode.models.deep_dense import DeepDenseNet
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datasets.p300_data import load_p300

from numpy.random import RandomState
rng = RandomState((2017,6,30))

    
def gaussian_perturbation(amps, rng):
    perturbation = rng.normal(0,0.0005,amps.shape).astype(np.float32)
    return perturbation

   
    

if __name__ =='__main__':  
    log = logging.getLogger(__name__)
    cuda = True                  
    acc=[]
    train = False
    save = False
    path = 'p300_sub1'
    for subject_id in range(1,9):
        model_name = '/home/al/braindecode/result/'+path+'/sub{:1d}'\
                   .format(subject_id)
        logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                            level=logging.DEBUG, stream=sys.stdout)
        
        n_classes=2
        X_y = load_p300( subject_id,n_classes,rng,shuffle=True)
        train_set = SignalAndTarget(X_y[0],X_y[1])
        train_set, test_set = split_into_two_sets(train_set,first_set_fraction=0.8)
        train_set, valid_set = split_into_two_sets(train_set,first_set_fraction=0.8)

        print train_set.X.shape
        set_random_seeds(seed=20190706, cuda=cuda)
        
            
        n_chans = train_set.X.shape[1]
        input_time_length = train_set.X.shape[2]
#        model = ShallowFBCSPNet(in_chans= n_chans,
#                     n_classes = n_classes,
#                     input_time_length= input_time_length,
#                     final_conv_length='auto',
#                ).create_network()
        
        model = DeepDenseNet(in_chans= n_chans,
                     n_classes = n_classes,
                     map_chans = 45,
                     input_time_length= input_time_length,
                     final_conv_length='auto',                               
                     bn_size=2,  ).create_network()
        model.cuda()    
        log.info("Model: \n{:s}".format(str(model)))
        optimizer = optim.Adam(model.parameters())
        iterator = BalancedBatchSizeIterator(batch_size=32)
        
        stop_criterion = Or([MaxEpochs(1000),
                             NoDecrease('valid_misclass', 100)])
        monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]
        model_constraint = MaxNormDefaultConstraint()
    
        if train==False:
            model= torch.load(model_name)
            exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                             loss_function=F.nll_loss, optimizer=optimizer,
                             model_constraint=model_constraint,
                             monitors=monitors,
                             stop_criterion=stop_criterion,
                             remember_best_column='valid_misclass',
                             run_after_early_stop=False, cuda=cuda)
            exp.run_eval()

        else:
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
        
#        if save:
#            path = 'p300_sub1'
#            model_name = '/home/al/braindecode/result/'+path+'/sub{:1d}'.format(subject_id)
#            torch.save(exp.model,model_name)
        