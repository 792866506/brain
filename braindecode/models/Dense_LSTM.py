#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:41:03 2017

@author: al
"""

import numpy as np
from torch import nn
import torch
from torch.nn import init
from torch.nn.functional import elu
import torch.nn.functional as F
from collections import OrderedDict
from braindecode.torch_ext.modules import Expression, AvgPool2dWithConv
from braindecode.torch_ext.util import np_to_var
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net

import  sys 
sys.path.insert(0,'/home/al/braindecode/code/braindecode/braindecode')
from models.eeg_densenet import EEGDenseNet
from models.eeg_resnet import EEGResNet
from models.deep_dense import DeepDenseNet
from models.CNNPlus import CNNPlus

def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:,:,:,0]
    if x.size()[2] == 1:
        x = x[:,:,0]
    #print("mean squeeze", th.mean(th.abs(x)))
    #assert False
    return x


def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)           


class Dense_LSTM(nn.Module): 
    def __init__(
                 self, 
                 in_chans,
                 n_classes,
                 num_conv_block = 3,
                 num_lstm_layers = 1,
                 final_conv_length='auto',
                 input_time_length=None,
                 pool_mode='max',
                 second_kernel_size=(2,32),
                 third_kernel_size=(8,4),
                 drop_prob=0.5
                 ):
        assert input_time_length % num_conv_block==0
        super(Dense_LSTM, self).__init__()
        if final_conv_length == 'auto':
            assert input_time_length is not None
        self.__dict__.update(locals())
        del self.self
        self.model_name = 'EEGNet'
        
        self.conv_block=[]
        branch = CNNPlus(in_chans=self.in_chans,
                         n_classes=self.n_classes,
                         input_time_length=self.input_time_length/self.num_conv_block
                              ).create_network()
        branch = branch.cuda()
        for i in xrange(self.num_conv_block):        
            new_model = nn.Sequential()
            for name, module in branch.named_children():
                if name == 'conv_classifier': 
                    break
                new_model.add_module(name, module)            
            self.conv_block.append(new_model)
        out = self.conv_block[0](np_to_var(np.ones(
                (1, self.in_chans, 
                 self.input_time_length/self.num_conv_block,1),
                dtype=np.float32)).cuda())            
        out = out.view(-1)
        input_size = out.size()[0]
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=128,
                            num_layers=self.num_lstm_layers,
                            bias=False,
                            batch_first=True,
                            dropout = 0.5,
                            )

        self.conv = nn.Conv2d(input_size,
                              64,
                              kernel_size=(3, 1),
                              stride=1,
                              )
        
        self.fc = nn.Sequential(
            nn.Linear(192, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(100, self.n_classes),
            nn.Softmax()
        )
        
          
    def forward(self, x):
        index = range(0,self.input_time_length,
                      self.input_time_length/self.num_conv_block)        
        branch_out = []        
        for i in xrange(len(index)):
            inputs = x[:,:,index[i]:index[i]+self.input_time_length/self.num_conv_block,:]                
            out = self.conv_block[i](inputs)            
            out = out.view(x.size()[0], 1, -1)            
            branch_out.append(out)        
        lstm_in = torch.cat((branch_out), 1)        
        lstm_out = self.lstm(lstm_in)
        lstm_out = lstm_out[0]
        lstm_out = lstm_out[:, -1, :]
        #print lstm_out.size()
        conv_in = lstm_in.permute(0, 2, 1).unsqueeze(3)  # 10, 2048, 7, 1
        #print conv_in.size()
        conv_out = self.conv(conv_in)  # 10, 64, 5, 1
        #print conv_out.size()
        conv_out = conv_out.view(x.size()[0], -1)  # 10, 320
        #print conv_out.size()

        fc_in = torch.cat([lstm_out, conv_out], 1)  # 10, 448
        #print fc_in.size()
        result = self.fc(fc_in)
        #print result.size()

        return result



if __name__ == '__main__':
    model = Dense_LSTM(in_chans=22,n_classes=4,input_time_length=1125).cuda()
    #title = t.autograd.Variable(t.arange(0,500).view(10,50)).long()
    #content = t.autograd.Variable(t.arange(0,2500).view(10,250)).long()
    X=np_to_var(np.random.randn(10,22,1125,1).astype('float32')).cuda()
    o = model(X)
    print(o.size())
