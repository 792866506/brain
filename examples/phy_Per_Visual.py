#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:40:50 2017

@author: al
"""
import mne
import numpy as np
from mne.io import concatenate_raws
from braindecode.datautil.signal_target import SignalAndTarget
from scipy import signal
from torch.nn.functional import elu
from torch.nn.functional import relu
# First 50 subjects as train
physionet_paths = [ mne.datasets.eegbci.load_data(sub_id,[4,8,12,]) for sub_id in range(1,2)]
physionet_paths = np.concatenate(physionet_paths)
parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto')
         for path in physionet_paths]

raw = concatenate_raws(parts)

picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epoched = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=1, tmax=4.1, proj=False, picks=picks,
                baseline=None, preload=True)


import pickle
pkl_file = open('/home/al/braindecode/data/phys/train_set.pkl', 'rb')
train_set = pickle.load(pkl_file)#  (2125, 64, 497)
pkl_file.close()

pkl_file = open('/home/al/braindecode/data/phys/test_set.pkl', 'rb')
test_set = pickle.load(pkl_file)
pkl_file.close()

from braindecode.models.deep4 import Deep4Net
from torch import nn
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model

# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = True
set_random_seeds(seed=20170629, cuda=cuda)

# This will determine how many crops are processed in parallel
input_time_length = 450
# final_conv_length determines the size of the receptive field of the ConvNet

model = Deep4Net(in_chans=64, n_classes=2, input_time_length=input_time_length,
                 filter_length_3=5, filter_length_4=5,
                 pool_time_stride=2,
                 stride_before_pool=True,
                        final_conv_length=1).create_network()
to_dense_prediction_model(model)

if cuda:
    model.cuda()

from torch import optim

optimizer = optim.Adam(model.parameters())

from braindecode.torch_ext.util import np_to_var
# determine output size
test_input = np_to_var(np.ones((2, 64, input_time_length, 1), dtype=np.float32))
if cuda:
    test_input = test_input.cuda()
out = model(test_input)
n_preds_per_input = out.cpu().data.numpy().shape[2]
print("{:d} predictions per input/trial".format(n_preds_per_input))

from braindecode.datautil.iterators import CropsFromTrialsIterator
iterator = CropsFromTrialsIterator(batch_size=32,input_time_length=input_time_length,
                                  n_preds_per_input=n_preds_per_input)


from braindecode.torch_ext.util import np_to_var, var_to_np
import torch.nn.functional as F
from numpy.random import RandomState
import torch as th
from braindecode.experiments.monitors import compute_preds_per_trial_for_set
rng = RandomState((2017,6,30))
'''
for i_epoch in range(50):
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

    # Print some statistics each epoch
    model.eval()
    for setname, dataset in (('Train', train_set),('Test', test_set)):
        # Collect all predictions and losses
        all_preds = []
        all_losses = []
        batch_sizes = []
        for batch_X, batch_y in iterator.get_batches(dataset, shuffle=False):
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(batch_y)
            if cuda:
                net_target = net_target.cuda()
            outputs = model(net_in)
            all_preds.append(var_to_np(outputs))
            outputs = th.mean(outputs, dim=2, keepdim=False)
            loss = F.nll_loss(outputs, net_target)
            loss = float(var_to_np(loss))
            all_losses.append(loss)
            batch_sizes.append(len(batch_X))
        # Compute mean per-input loss
        loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
                       np.mean(batch_sizes))
        print("{:6s} Loss: {:.5f}".format(setname, loss))
        preds_per_trial = compute_preds_per_trial_for_set(all_preds,
                                                          input_time_length,
                                                          dataset)
        # preds per trial are now trials x classes x timesteps/predictions
        # Now mean across timesteps for each trial to get per-trial predictions
        meaned_preds_per_trial = np.array([np.mean(p, axis=1) for p in preds_per_trial])
        predicted_labels = np.argmax(meaned_preds_per_trial, axis=1)
        accuracy = np.mean(predicted_labels == dataset.y)
        print("{:6s} Accuracy: {:.1f}%".format(
            setname, accuracy * 100))
'''
model.load_state_dict(th.load('/home/al/braindecode/phy.pkl'))        
        
train_batches = list(iterator.get_batches(train_set, shuffle=False))
train_X_batches = np.concatenate(list(zip(*train_batches))[0])




new_model = nn.Sequential()
for name, module in model.named_children():
    if name == 'softmax': break
    new_model.add_module(name, module)

new_model.eval();
pred_fn = lambda x: var_to_np(th.mean(new_model(np_to_var(x).cuda())[:,:,:,0], dim=2, keepdim=False))
#pred_fn = lambda x: var_to_np(new_model(np_to_var(x).cuda()))

from braindecode.visualization.perturbation import compute_correlations
from braindecode.visualization.perturbation import compute_amplitude_prediction_correlations
import logging
import sys
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)

fs = epoched.info['sfreq']
nperseg=600
'''
amp_pred_corrs = compute_correlations(pred_fn, train_X_batches, n_iterations=11,
                                      fs=fs,nperseg=nperseg,
                                         batch_size=10)
'''
amp_pred_corrs = compute_amplitude_prediction_correlations(pred_fn, train_X_batches, n_iterations=11,
                                              batch_size=10)


'''
abs_amp_pred_corrs = np.abs(amp_pred_corrs)

mean_corr = np.mean(abs_amp_pred_corrs,axis=(0,1,3))
plt.figure(1)
plt.plot(freqs,mean_corr)###########

mean_corr = np.mean(amp_pred_corrs, axis=0)
mean_corr = np.mean(mean_corr, axis=(0,2))
mean_corr= np.abs(mean_corr)
plt.figure(2)
plt.plot(freqs,mean_corr)
'''
mean_corr = np.mean(amp_pred_corrs, axis=0)


freqs = np.fft.rfftfreq(train_X_batches.shape[2], d=1.0/fs)
#freqs, times, _ = signal.stft(train_X_batches[0,:,:,0], fs=fs, nperseg=nperseg)
start_freq = 7
stop_freq = 14

i_start = np.searchsorted(freqs,start_freq)
i_stop = np.searchsorted(freqs, stop_freq) + 1

freq_corr = np.mean(mean_corr[:,i_start:i_stop], axis=1)

from braindecode.datasets.sensor_positions import get_channelpos, CHANNEL_10_20_APPROX

ch_names = [s.strip('.') for s in epoched.ch_names]
positions = [get_channelpos(name, CHANNEL_10_20_APPROX) for name in ch_names]
positions = np.array(positions)

import matplotlib.pyplot as plt
from matplotlib import cm

max_abs_val = np.max(np.abs(freq_corr))
'''
fig, axes = plt.subplots(2, len(times))
class_names = ['Left Hand', 'Right Hand']
for t in range(len(times)):
    for i_class in range(2):
        ax = axes[i_class,t]
        mne.viz.plot_topomap(freq_corr[:,t,i_class], positions,
                         vmin=-max_abs_val, vmax=max_abs_val, contours=0,
                        cmap=cm.coolwarm, axes=ax, show=False);
        ax.set_title(class_names[i_class])

from braindecode.visualization.plot import ax_scalp

fig, axes = plt.subplots(2, len(times))
class_names = ['Left Hand', 'Right Hand']
for t in range(len(times)):
    for i_class in range(2):
        ax = axes[i_class,t]
        ax_scalp(freq_corr[:,t,i_class], ch_names, chan_pos_list=CHANNEL_10_20_APPROX, cmap=cm.coolwarm,
            vmin=-max_abs_val, vmax=max_abs_val, ax=ax)
        ax.set_title(class_names[i_class])
    
    
    
'''
fig, axes = plt.subplots(1, 2)
class_names = ['Left Hand', 'Right Hand']
for i_class in range(2):
    ax = axes[i_class]
    mne.viz.plot_topomap(freq_corr[:,i_class], positions,
                     vmin=-max_abs_val, vmax=max_abs_val, contours=0,
                    cmap=cm.coolwarm, axes=ax, show=False);
    ax.set_title(class_names[i_class])
