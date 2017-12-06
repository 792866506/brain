import logging
import os.path
import time
from collections import OrderedDict
import sys

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
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne


import  sys
sys.path.insert(0,'/home/al/braindecode/code/braindecode/braindecode')
from models.eeg_densenet import EEGDenseNet
from models.eeg_resnet import EEGResNet

log = logging.getLogger(__name__)


def run_exp(data_folder, subject_id, low_cut_hz, model, cuda, exp_std ,pca):
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

    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),])
                           #   ('Foot', [3]), ('Tongue', [4])])
    ival = [-500, 4000]

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)

    train_set, valid_set = split_into_two_sets(train_set,
                                               first_set_fraction=0.8)

    set_random_seeds(seed=20190706, cuda=cuda)

    n_classes = 2
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
                 first_filter_length=5,
                 nonlinearity=elu,
                 split_first_layer=True,
                 batch_norm_alpha=0.1,
                 growth_rate=20, 
                 bn_size=8, 
                 drop_rate=0.5, 
                 block_config=(2,2,2,2,2),#2 2 2 2 2 2   50
                 compression=0.5,
                 num_init_features=25, 
                 ).create_network()
        
    if cuda:
        model.cuda()
    log.info("Model: \n{:s}".format(str(model)))

    optimizer = optim.Adam(model.parameters())

    iterator = BalancedBatchSizeIterator(batch_size=60)

    stop_criterion = Or([MaxEpochs(100),
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
    return exp

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
    # Should contain both .gdf files and .mat-labelfiles from competition
    data_folder = '/home/al/BCICIV_2a_gdf/'
    subject_id = 1 # 1-9
    low_cut_hz = 4# 0 or 4
    model = 'deep' #'shallow' or 'deep'
    cuda = True
    exp_std =True
    pca=False
    exp = run_exp(data_folder, subject_id, low_cut_hz, model, cuda,exp_std,pca)
    log.info("Last 10 epochs")
    log.info("\n" + str(exp.epochs_df.iloc[-10:]))
    
'''    
482       0.204861  0.381766  
483       0.190972  0.393086  
484       0.190972  0.397047  
485       0.201389  0.392911  
486       0.211806  0.393300  
487       0.215278  0.391213  
488       0.218750  0.394214  
489       0.229167  0.386034  
490       0.215278  0.381247  
491       0.204861  0.379195  
'''