#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 22:51:03 2017

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

class EEGNet(nn.Module): 
    def __init__(
                 self, 
                 in_chans,
                 n_classes,
                 final_conv_length='auto',
                 input_time_length=None,
                 pool_mode='max',
                 second_kernel_size=(2,32),
                 third_kernel_size=(8,4),
                 drop_prob=0.25
                 ):
        
        super(EEGNet, self).__init__()
        if final_conv_length == 'auto':
            assert input_time_length is not None
        self.model_name = 'EEGNet'
        self.n_classes=n_classes
        #self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)
        self.conv_time=nn.Sequential(
                                     Expression(LAMBDA_X1),
                                     nn.Conv2d(1, 1,
                                                    (448, 1),
                                                    stride=1, ),
                                     nn.ReLU(),
                                     Expression(LAMBDA_X2),
                                     Expression(squeeze_output)
                                     
                                     )
        self.title_lstm = nn.LSTM(input_size = 64,\
                            hidden_size =256,
                            num_layers = opt.num_layers,
                            bias = True,
                            batch_first = False,
                            # dropout = 0.5,
                            bidirectional = True
                            )


        # self.dropout = nn.Dropout()
        self.fc = nn.Sequential(
                              
            #nn.Linear(1536,opt.linear_hidden_size),
            #nn.BatchNorm1d(opt.linear_hidden_size),
            #nn.ReLU(inplace=True),
            
            nn.Linear(1536,1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),

            nn.Linear(1000,2),

            
        )
        
        # self.fc = nn.Linear(3 * (opt.title_dim+opt.content_dim), opt.num_classes)
        #if opt.embedding_path:
            #self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))

    def forward(self, X):
        conv_out=self.conv_time(X)
        if self.print_==True:
            print conv_out.data.shape
            #[10, 50, 64]
            
        title_out = self.title_lstm(conv_out.permute(1,0,2))[0].permute(1,2,0) 
        if self.print_==True:
            print title_out.data.shape
            #[10, 512, 50]
        #content_out = self.content_lstm(content.permute(1,0,2))[0].permute(1,2,0)


        title_conv_out = kmax_pooling((title_out),2,self.opt.kmax_pooling)
        if self.print_==True:
            print title_conv_out.data.shape
            #[10, 512, 3]
        #content_conv_out = kmax_pooling((content_out),2,self.opt.kmax_pooling)
        
            
        #conv_out = t.cat((title_conv_out,content_conv_out),dim=1)
        reshaped = title_conv_out.view(title_conv_out.size(0), -1)
        
        if self.print_==True:
            print reshaped.data.shape
            #[10, 1536]
        #conv_out = t.cat((reshaped,content_conv_out),dim=1)
        conv_out=reshaped
        if self.print_==True:
            print conv_out.data.shape#[10, 1736]
        logits = self.fc((conv_out))
        if self.print_==True:
            print logits.data.shape#[10, 1999]
        return logits

    # def get_optimizer(self):  
    #    return  t.optim.Adam([
    #             {'params': self.title_conv.parameters()},
    #             {'params': self.content_conv.parameters()},
    #             {'params': self.fc.parameters()},
    #             {'params': self.encoder.parameters(), 'lr': 5e-4}
    #         ], lr=self.opt.lr)
    # # end method forward


 
if __name__ == '__main__':
    from config import opt
    m = LSTMEEG(opt)
    #title = t.autograd.Variable(t.arange(0,500).view(10,50)).long()
    #content = t.autograd.Variable(t.arange(0,2500).view(10,250)).long()
    X=t.autograd.Variable(t.randn(10,64,497,1))
    o = m(X)
    print(o.size())

