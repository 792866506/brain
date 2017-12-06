#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:13:50 2017

@author: al
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



class EEGDenseNet(object):
    """
    Dense Network for EEG.
    """
    def __init__(self, in_chans,
                 n_classes,
                 input_time_length,
                 final_pool_length='auto',
                 first_filter_length=3,
                 nonlinearity=elu,
                 split_first_layer=True,
                 batch_norm_alpha=0.1,
                 growth_rate=50, 
                 bn_size=4, 
                 drop_rate=0.5, 
                 block_config=(2, 2, 3 ,3 ,4,4),#2 2 2 2 2 2   50
                 compression=0.5,
                 num_init_features=24, 
                 ):
        if final_pool_length == 'auto':
            assert input_time_length is not None
        assert first_filter_length % 2 == 1
        self.__dict__.update(locals())
        del self.self

    def create_network(self):
        self.n_first_filters = self.num_init_features
        block_config = self.block_config
        bn_size = self.bn_size
        growth_rate = self.growth_rate
        drop_rate = self.drop_rate
        compression =self.compression
        model = nn.Sequential()
        if self.split_first_layer:
            model.add_module('dimshuffle', Expression(_transpose_time_to_spat))
            model.add_module('conv_time', nn.Conv2d(1, self.n_first_filters,
                                                    (
                                                    self.first_filter_length, 1),
                                                    stride=1,
                                                    padding=(self.first_filter_length // 2, 0)))
            model.add_module('conv_spat',
                             nn.Conv2d(self.n_first_filters, self.n_first_filters,
                                       (1, self.in_chans),
                                       stride=(1, 1),
                                       bias=False))
        else:
            model.add_module('conv_time',
                             nn.Conv2d(self.in_chans, self.n_first_filters,
                                       (self.first_filter_length, 1),
                                       stride=(1, 1),
                                       padding=(self.first_filter_length // 2, 0),
                                       bias=False,))
        n_filters_conv = self.n_first_filters
        model.add_module('bnorm',
                         nn.BatchNorm2d(n_filters_conv,
                                        momentum=self.batch_norm_alpha,
                                        affine=True,
                                        eps=1e-5),)
        model.add_module('conv_nonlin', Expression(self.nonlinearity))
        
        num_features = self.num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            model.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features
                                                            * compression))
                model.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        model.add_module('norm5', nn.BatchNorm2d(num_features))
        model.add_module('elu',nn.ELU( inplace=True))
        model.eval()
        if self.final_pool_length == 'auto':
            out = model(np_to_var(np.ones(
                (1, self.in_chans, self.input_time_length,1),
                dtype=np.float32)))
            #print out.size()
            n_cur_filters = out.size()[1]
            n_out_time = out.size()[2]
            self.final_pool_length = n_out_time
        else :
            n_cur_filters = num_features
        model.add_module('mean_pool', nn.MaxPool2d(
            (self.final_pool_length, 1), (1,1))
            )
        model.add_module('conv_classifier',
                             nn.Conv2d(n_cur_filters, self.n_classes,
                                       (1, 1), bias=True))
        model.add_module('softmax', nn.LogSoftmax())
        model.add_module('squeeze',  Expression(_squeeze_final_output))

        # Initialization, xavier is same as in our paper...
        # was default from lasagne
        init.kaiming_normal(model.conv_time.weight, a=0)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer:
            init.constant(model.conv_time.bias, 0)
            init.kaiming_normal(model.conv_spat.weight, a=0)

        init.constant(model.bnorm.weight, 1)
        init.constant(model.bnorm.bias, 0)
        
        # Residual Block initialization happens already in ResidualBlock
        '''
        init.constant(model.norm5.weight, 1)
        init.constant(model.norm5.bias, 0)
        '''
        init.kaiming_normal(model.conv_classifier.weight, a=0)
        init.constant(model.conv_classifier.bias, 0)

        # Start in eval mode
        model.eval()
        return model


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


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, 
                 drop_rate, conv_length = 3 ):
        super(_DenseLayer, self).__init__()
        self.add_module('bn1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu_1', nn.ELU(inplace=True)),
        self.add_module('conv_1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('bn2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu_2', nn.ELU(inplace=True)),
        self.add_module('conv_2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=(conv_length,1), stride=1, 
                        padding=((conv_length-1)/2,0), bias=False)),
        self.drop_rate = drop_rate
        
        '''
        for conv_layer in (self.conv_1, self.conv_2):
            init.kaiming_normal(conv_layer.weight, a=0)
            #init.xavier_uniform(conv_layer.weight, gain=1)
            #init.constant(conv_layer.bias, 0)
        for bn_layer in (self.bn1, self.bn2):
            init.constant(bn_layer.weight, 1)
            init.constant(bn_layer.bias, 0)
        '''
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ELU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=(2,1), stride=(2,1)))
        '''
        init.kaiming_normal(self.conv.weight, a=0)
        init.constant(self.norm.weight, 1)
        init.constant(self.norm.bias, 0)
        '''
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
aa=EEGDenseNet(64,2,497).create_network()