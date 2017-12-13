#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 18:25:41 2017

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


def _transpose(x):
    return x.permute(0, 2, 1, 3)  
      

class RCNN_EEG(nn.Module): 
    def __init__(self,  in_chans,
                 n_classes,
                 input_time_length=None,
                 n_filters_time=30,
                 filter_time_length=11,
                 n_filters_spat=40,
                 pool_time_length=75,
                 pool_time_stride=15,
                 final_conv_length=30,
                 split_first_layer=True,
                 batch_norm=True,
                 batch_norm_alpha=0.1,
                 drop_prob=0.5 ):
        super(RCNN_EEG, self).__init__()
        self.__dict__.update(locals())
        del self.self
        
        self.model_name = 'RCNN_EEG'
        kernel_size = 3
        
        model = DeepDenseNet(in_chans= self.in_chans,
                     n_classes = self.n_classes,
                     input_time_length= self.input_time_length,
                     final_conv_length='auto',                               
                     bn_size=2,  ).create_network()
        self.conv_time=nn.Sequential( )        
        self.conv_time.add_module('conv', 
                                  nn.Conv2d(self.in_chans, self.n_filters_time,
                                        ( self.filter_time_length, 1),
                                        stride=1, ))
        self.conv_time.add_module('pool',nn.MaxPool2d(kernel_size=(3,1),
                                        stride=(3,1)))
        self.conv_time.add_module('bn',nn.BatchNorm2d(self.n_filters_time))
        self.conv_time.add_module('relu',nn.ReLU(inplace=True),) 
        self.conv_time.add_module('dimshuffle',Expression(_transpose))
        self.conv_time.add_module('squeeze',Expression(_squeeze_final_output))
        
        out =self.conv_time(np_to_var(np.ones(
                (1, self.in_chans, self.input_time_length,1),
                dtype=np.float32)))
        input_size =  out.size()[2]
        dummy_input = out
        print dummy_input.size()                    
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = 48,
                            num_layers = 2,
                            bias = True,
                            batch_first = True,
                            dropout = 0.5,
                            bidirectional = True
                            )
        lstm_out = self.lstm(dummy_input)
        lstm_out = lstm_out[0]
        lstm_out = lstm_out[:, -1, :]
        print lstm_out.size()
        fc_input_size = lstm_out.view(-1).size()[0]
        self.fc = nn.Sequential( OrderedDict([
                ('drop1',nn.Dropout(0.5)),
                ('linear1',nn.Linear(fc_input_size,50)),
                ('bn1',nn.BatchNorm1d(50)),
                ('relu1',nn.ReLU(inplace=True)),
                ('classifier',nn.Linear(50,self.n_classes)),
                ('softmax',nn.Softmax()),
                ]))              
        init.xavier_uniform(self.conv_time.conv.weight, gain=1)    
        init.constant(self.conv_time.conv.bias, 0)
        
        init.xavier_uniform(self.fc.linear1.weight, gain=1)
        #init.xavier_uniform(self.fc.linear2.weight, gain=1)
        
        
        init.constant(self.fc.linear1.bias, 0)
        #init.constant(self.fc.linear2.bias, 0)
                
    def forward(self, X): 
        out=self.conv_time(X)  
        #print out.size()
        out = self.lstm(out)[0]
        out= torch.mean(out,1)
        #print out.size()            
        out = self.fc(out)
        return out



 
if __name__ == '__main__':
    m = RCNN_EEG(22,4,1125)
    X=np_to_var(np.random.randn(10,22,1125,1).astype('float32'))
    o = m(X)
    print(o.size())
