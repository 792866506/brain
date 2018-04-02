#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:45:17 2018

@author: al
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:48:39 2017

@author:    al
"""


import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch as th
from torch.nn.functional import elu
import torch.nn.functional as F
from collections import OrderedDict

from braindecode.torch_ext.modules import Expression, AvgPool2dWithConv
from braindecode.torch_ext.functions import identity
from braindecode.torch_ext.util import np_to_var

import sys
sys.path.insert(0,'/home/al/braindecode/code/braindecode/braindecode')


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight, gain=1)
    elif isinstance(m,nn.BatchNorm2d):
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)
        

class DeepDenseNet(nn.Sequential):
    """
    Dense Network for EEG.
    """
    def __init__(self, in_chans,
                 n_classes,
                 input_time_length,
                 map_chans = 30,
                 n_first_filters = 25,
                 final_conv_length='auto',
                 first_filter_length=11,
                 pool_time_length=3,
                 pool_time_stride=3,
                 nonlinearity=elu,
                 split_first_layer=True,
                 batch_norm_alpha=0.1,
                 bn_size=2, 
                 drop_rate=0.5,):
        super(DeepDenseNet,self).__init__()
        if final_conv_length == 'auto':
            assert input_time_length is not None
        assert first_filter_length % 2 == 1
        self.__dict__.update(locals())
        del self.self
       
        self.add_module('dimshuffle0', Expression(_transpose1))
        #   2,1,22,1125  ->  2,22,1125.1
        self.add_module('fc',nn.Conv2d(self.in_chans,self.map_chans,(1,1),1,bias=False))
        self.add_module('bn_fc',nn.BatchNorm2d(self.map_chans))

        if self.split_first_layer:
            self.add_module('dimshuffle1', Expression(_transpose_time_to_spat))#2,1,1125,22
            self.add_module('conv_time', nn.Conv2d(1, self.n_first_filters,
                                                    (
                                                    self.first_filter_length, 1),
                                                    stride=1,
                                                    #dilation=(2,1),
                                                    ))
            self.add_module('conv_spat',
                             nn.Conv2d(self.n_first_filters, self.n_first_filters,
                                       (1, 30),
                                       stride=(1, 1),
                                       bias=False))
            
            
        else:
            self.add_module('conv_time',
                             nn.Conv2d(self.in_chans, self.n_first_filters,
                                       (self.first_filter_length, 1),
                                       stride=(1, 1),
                                       padding=(self.first_filter_length // 2, 0),
                                       bias=False,))
        self.add_module('bn',
                             nn.BatchNorm2d(self.n_first_filters,
                                            momentum=self.batch_norm_alpha,
                                            affine=True,
                                            eps=1e-5),)
        self.add_module('elu', nn.ELU(inplace=True))
        self.add_module('pool',
                         nn.MaxPool2d(
                             kernel_size=(self.pool_time_length, 1),stride=(3, 1)))
                             #stride=(self.pool_time_length, 1)))
                       
                        
                        
        self.add_module('drop2',nn.Dropout(inplace=True
                             ))                     
        n_filters_conv = self.n_first_filters
        self.add_module('2_DenseLayer',_DenseLayer(n_filters_conv,50,bn_size, 
                 drop_rate,  ))
        self.add_module('pool2',nn.MaxPool2d(
                             kernel_size=(self.pool_time_length, 1),stride=(3, 1)))
        
        
        
        self.add_module('drop3',nn.Dropout(inplace=True))
        self.add_module('3_DenseLayer',_DenseLayer(n_filters_conv+50,100,bn_size, 
                 drop_rate,  ))
        self.add_module('pool3', nn.MaxPool2d(kernel_size=(3,1), stride=(3,1)))
        
        
        
        self.add_module('drop4',nn.Dropout(p=0.5,inplace=True))
        self.add_module('4_Transition',
                         _Transition(175, 100))
        
        self.eval()
        if self.final_conv_length == 'auto':
            out = self(np_to_var(np.ones(
                (1, 1,self.in_chans, self.input_time_length),
                dtype=np.float32)))
            print out.size()
            n_cur_filters = out.size()[1]
            n_out_time = out.size()[2]
            self.final_conv_length = n_out_time
        else :
            n_cur_filters = 100
            self.final_conv_length = 8
            
            
        self.add_module('conv_classifier',
                             nn.Conv2d(n_cur_filters, self.n_classes,
                                       (self.final_conv_length,out.size()[3] ), bias=True))

        self.add_module('squeeze',  Expression(_squeeze_final_output))
        self.add_module('softmax', nn.LogSoftmax(dim=-1))
        
        # Initialization, xavier is same as in our paper...
        # was default from lasagne
        self.apply(weights_init)



# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
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

def _transpose1(x):
    return x.permute(0, 2,3, 1)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, 
                 drop_rate, conv_size = (11,1)):
        super(_DenseLayer, self).__init__()
        #self.add_module('drop',nn.Dropout(inplace=True))
        self.add_module('conv_1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('bn1', nn.BatchNorm2d(bn_size * growth_rate, affine=True,)),
        self.add_module('elu_1', nn.ELU(inplace=True)),
        #self.add_module('drop2',nn.Dropout(inplace=True))
        self.add_module('conv_2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=conv_size, stride=1, 
                        #dilation=(3,1),
                        padding=((conv_size[0]-1)/2,(conv_size[1]-1)/2), bias=False)),
        self.add_module('bn2', nn.BatchNorm2d(growth_rate, affine=True),),
        self.add_module('elu_2', nn.ELU(inplace=True)),
        self.drop_rate = drop_rate
        '''
        self.add_module('selayer',
                        SELayer(channel=growth_rate,reduction=growth_rate//8))
        '''

        
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        #if self.drop_rate > 0:
            #new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        #print new_features.shape
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=(11,1), 
                                          #dilation=(2,1),
                                          stride=1, bias=False))
        self.add_module('bn', nn.BatchNorm2d(num_output_features, affine=True,))
        self.add_module('elu', nn.ELU(inplace=True))
        self.add_module('pool', nn.MaxPool2d(kernel_size=(3,1), stride=(3,1)))


        
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
            
if __name__ == '__main__':    
    model=DeepDenseNet(22,4,1125)
    out=model((np_to_var(np.ones((2,1,22,1125)),dtype='float32')))