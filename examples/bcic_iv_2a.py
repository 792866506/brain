import logging
import os.path
import time
from collections import OrderedDict


import numpy as np
import torch
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
from braindecode.datautil.signal_target import SignalAndTarget
log = logging.getLogger(__name__)

#from tensorboardX import SummaryWriter
import sys
sys.path.insert(0,'/home/al/braindecode/code/braindecode/braindecode')
from models.deep_dense import DeepDenseNet
from models.depth_dense  import DepthDenseNet
from models.resnet  import EEGResNet


#ival = [100, 4600]
def run_exp(data_folder, subject_id, low_cut_hz,high_cut_hz, model_name, cuda,pca=False):
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

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)
     
    train_set, valid_set = split_into_two_sets(train_set,
                                               first_set_fraction=0.8)

    
    print train_set.X.shape
    ####
 
    set_random_seeds(seed=20190706, cuda=cuda)

    n_classes = len(marker_def)
    n_chans = 22
    input_time_length = 1125
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
                     bn_size=1,  ).create_network()
        
      
    if cuda:
        model.cuda()    
    log.info("Model: \n{:s}".format(str(model)))

    optimizer = optim.Adam(model.parameters())

    iterator = BalancedBatchSizeIterator(batch_size=64)

    stop_criterion = Or([MaxEpochs(1000),
                         NoDecrease('valid_misclass', 100)])
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

def test_exp(data_folder, subject_id, low_cut_hz,high_cut_hz, model_name, cuda,
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
          
    train_set, valid_set = split_into_two_sets(train_set,
                                               first_set_fraction=0.8)

    set_random_seeds(seed=20190706, cuda=cuda)

    
    model = torch.load(model_name)

        
    
    if cuda:
        model.cuda()    
    log.info("Model: \n{:s}".format(str(model)))

    optimizer = optim.Adam(model.parameters())

    iterator = BalancedBatchSizeIterator(batch_size=500)
    '''
    stop_criterion = Or([MaxEpochs(1600),
                         NoDecrease('valid_misclass', 160)])
    '''
    stop_criterion = Or([MaxEpochs(1000),
                         NoDecrease('valid_misclass', 100)])
    monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]

    model_constraint = MaxNormDefaultConstraint()
    loss_function = F.nll_loss
    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=loss_function, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=True, cuda=cuda)
   
    exp.run_eval()
    return exp



if __name__ == '__main__':
    mean=[]
    mini=[]
    confuse_mat =[]
    train = False              
    path='shallow_0001'  
    ival = [-500, 4000]
    for subject_id in xrange(1,10):
        logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                            level=logging.DEBUG, stream=sys.stdout)
        # Should contain both .gdf files and .mat-labelfiles from competition
        data_folder = '/home/al/BCICIV_2a_gdf/'
        low_cut_hz = 0 # 0 or 4
        high_cut_hz = 38
        model_name = 'deep' #'shallow' or 'deep'  cnn++  Dense_LSTM  RCNN_EEG  resnet
        cuda = True
        if train:
            exp = run_exp(data_folder, subject_id, low_cut_hz,high_cut_hz, model_name, cuda)
            log.info("Last 10 epochs")
            log.info("\n" + str(exp.epochs_df.iloc[-10:]))
            log.info("\n" + str(1-exp.epochs_df['test_misclass'].iloc[-10:]))          
            torch.save(exp.model,'/home/al/braindecode/result/'+path+'/sub{:01d}'.format(subject_id))
        
        
        else :
            model_name = '/home/al/braindecode/result/'+path+'/sub{:01d}'.format(subject_id)
            #deep_dense_model   shallow_model  se_dense  se_32_16  dense_enter_se
            exp = test_exp(data_folder, subject_id, low_cut_hz,high_cut_hz, model_name, cuda)
            log.info("\n" + str(exp.epochs_df))
            log.info("\n" + str(1-exp.epochs_df['test_misclass'].iloc[-10:]))
            
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
#np.save('/home/al/braindecode/result/'+path+'/confuse_mat.npy',new_confuse_mat)
'''
