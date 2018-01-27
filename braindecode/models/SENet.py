#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:26:23 2017

@author: al
"""

import numpy as np
from torch import nn
from torch.nn import init
from torch.nn.functional import elu
from collections import OrderedDict
from braindecode.torch_ext.modules import Expression, AvgPool2dWithConv
from braindecode.torch_ext.functions import identity
from braindecode.torch_ext.util import np_to_var
'''
def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:,:,:,0]
    if x.size()[2] == 1:
        x = x[:,:,0]
    #print("mean squeeze", th.mean(th.abs(x)))
    #assert False
    return x
class RCNN_EEG(nn.Module): 
    def __init__(self,  in_chans,in_size,out_chans,
                  batch_norm=True,
                  drop_prob=0.5 ):
        super(RCNN_EEG, self).__init__()
        self.__dict__.update(locals())
        del self.self
        height = self.in_size[2]
        width = self.in_size[3]
        mid_channle = in_chans//8
        self.excitation = nn.Sequential( OrderedDict([
                        ('global_avepool',nn.AvgPool2d((height,width) )),
                        ('squeeze',Expression(_squeeze_final_output)),
                        ('fc1',nn.Linear(self.in_chans,mid_channle)),
                        ('bn1',nn.BatchNorm1d(mid_channle)),
                        ('relu1',nn.ReLU(inplace=True)),
                        ('fc2',nn.Linear(mid_channle,self.out_chans)),
                        #('bn2',nn.BatchNorm1d(self.out_chan)),
                        #('relu2',nn.ReLU(inplace=True)),
                        ('sigmoid',nn.Sigmoid())
                        ]))

    def forward(self,x):
        out = self.excitation(x)
        out = out
        
from torch import nn
'''

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(OrderedDict([
                ('drop',nn.Dropout()),
                ('fc1',nn.Linear(channel, reduction)),
                ('relu1',nn.ReLU(inplace=True)),
                ('fc2',nn.Linear(reduction, channel)),
                ('sigmoid',nn.Sigmoid())
        ]))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y