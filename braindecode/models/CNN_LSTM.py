#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:31:09 2017

@author: al
"""

import torch as t
import torch
import numpy as np
from torch import nn
from torch.nn import init

from collections import OrderedDict


class CNN_LSTM(nn.Module):
    def __init__(
            self,
            relu=True,
            batch_norm=True,
            in_chans=3,
            num_classes=4,
            batch_norm_alpha=0.1,
            num_filters_1=32,
            filter_length_1=3,

            num_filters_2=64,
            filter_length_2=3,

            num_filters_3=128,
            filter_length_3=3,

            num_time_windows=7,

            num_lstm_layers=1,
            drop_prob=0.5,
    ):

        super(CNN_LSTM, self).__init__()

        self.__dict__.update(locals())
        del self.self

        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.in_chans,
                                self.num_filters_1,
                                kernel_size=self.filter_length_1,
                                stride=1,
                                padding=(1, 1))),
            ('batch_norm_1', nn.BatchNorm2d(self.num_filters_1,
                                            momentum=self.batch_norm_alpha,
                                            affine=True,
                                            eps=1e-5)),
            ('relu1', nn.ReLU(inplace=True)),
            
            ('drop2',nn.Dropout(p=self.drop_prob)),
            ('conv2', nn.Conv2d(self.num_filters_1,
                                self.num_filters_1,
                                kernel_size=self.filter_length_1,
                                stride=1,
                                padding=(1, 1))),
            ('batch_norm_2', nn.BatchNorm2d(self.num_filters_1,
                                            momentum=self.batch_norm_alpha,
                                            affine=True,
                                            eps=1e-5)),
            ('relu2', nn.ReLU(inplace=True)),

            
            ('drop3',nn.Dropout(p=self.drop_prob)),
            ('conv3', nn.Conv2d(self.num_filters_1,
                                self.num_filters_1,
                                kernel_size=self.filter_length_1,
                                stride=1,
                                padding=(1, 1))),
            ('batch_norm_3', nn.BatchNorm2d(self.num_filters_1,
                                            momentum=self.batch_norm_alpha,
                                            affine=True,
                                            eps=1e-5)),
            ('relu3', nn.ReLU(inplace=True)),

            
            ('drop4',nn.Dropout(p=self.drop_prob)),
            ('conv4', nn.Conv2d(self.num_filters_1,
                                self.num_filters_1,
                                kernel_size=self.filter_length_1,
                                stride=1,
                                padding=(1, 1))),
            ('batch_norm_4', nn.BatchNorm2d(self.num_filters_1,
                                            momentum=self.batch_norm_alpha,
                                            affine=True,
                                            eps=1e-5)),
            ('relu4', nn.ReLU(inplace=True)),

            ('max_pool_1', nn.MaxPool2d(kernel_size=(2, 2),
                                        stride=(2, 2))
             ),

            
            ('drop5',nn.Dropout(p=self.drop_prob)),
            ('conv5', nn.Conv2d(self.num_filters_1,
                                self.num_filters_2,
                                kernel_size=self.filter_length_2,
                                stride=1,
                                padding=(1, 1))),
            ('batch_norm_5', nn.BatchNorm2d(self.num_filters_2,
                                            momentum=self.batch_norm_alpha,
                                            affine=True,
                                            eps=1e-5)),
            ('relu5', nn.ReLU(inplace=True)),

            
            ('drop6',nn.Dropout(p=self.drop_prob)),
            ('conv6', nn.Conv2d(self.num_filters_2,
                                self.num_filters_2,
                                kernel_size=self.filter_length_2,
                                stride=1,
                                padding=(1, 1))),
            ('batch_norm_6', nn.BatchNorm2d(self.num_filters_2,
                                            momentum=self.batch_norm_alpha,
                                            affine=True,
                                            eps=1e-5)),
            ('relu6', nn.ReLU(inplace=True)),

            ('max_pool_2', nn.MaxPool2d(kernel_size=(2, 2),
                                        stride=(2, 2))
             ),

            
            ('drop7',nn.Dropout(p=self.drop_prob)),
            ('conv7', nn.Conv2d(self.num_filters_2,
                                self.num_filters_3,
                                kernel_size=self.filter_length_3,
                                stride=1,
                                padding=(1, 1))),
            ('batch_norm_7', nn.BatchNorm2d(self.num_filters_3,
                                            momentum=self.batch_norm_alpha,
                                            affine=True,
                                            eps=1e-5)),
            ('relu7', nn.ReLU(inplace=True)),

            ('max_pool_3', nn.MaxPool2d(kernel_size=(2, 2),
                                        stride=(2, 2))
             ),

        ]))
        conv_list=[self.branch1.conv1,
                   self.branch1.conv2,
                   self.branch1.conv3,
                   self.branch1.conv4,
                   self.branch1.conv5,
                   self.branch1.conv6,
                   self.branch1.conv7
                   ]
        for module in conv_list:
            init.xavier_uniform(module.weight, gain=1)
            init.constant(module.bias, 0)

        self.branch2 = nn.Sequential()
        for name, module in self.branch1.named_children():
            self.branch2.add_module(name, module)

        self.branch3 = nn.Sequential()
        for name, module in self.branch1.named_children():
            self.branch3.add_module(name, module)

        self.branch4 = nn.Sequential()
        for name, module in self.branch1.named_children():
            self.branch4.add_module(name, module)

        self.branch5 = nn.Sequential()
        for name, module in self.branch1.named_children():
            self.branch5.add_module(name, module)

        self.branch6 = nn.Sequential()
        for name, module in self.branch1.named_children():
            self.branch6.add_module(name, module)

        self.branch7 = nn.Sequential()
        for name, module in self.branch1.named_children():
            self.branch7.add_module(name, module)

        test_input = torch.autograd.Variable(torch.ones(1, 3, 32, 32))
        test_out = self.branch1(test_input)
        # print test_out.size()
        test_out = test_out.view(-1)
        input_size = test_out.size()[0]
        #print input_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=128,
                            num_layers=self.num_lstm_layers,
                            bias=True,
                            batch_first=True,
                            # dropout = 0.5,

                            )

        self.conv = nn.Conv2d(input_size,
                              64,
                              kernel_size=(3, 1),
                              stride=1,
                              )

        self.fc = nn.Sequential(
            nn.Linear(448, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.num_classes),
            nn.Softmax()
        )
        
        
        

    def forward(self, x):
        branch_out = []
        # print 'aaaaaaaaaaaaaaaaa'
        model_list = [self.branch1, self.branch2, self.branch3, self.branch4, self.branch5,
                      self.branch6, self.branch7]
        for i in range(self.num_time_windows):
            out = model_list[i](x[:,i])
            out = out.view(x.size()[0], 1, -1)
            branch_out.append(out)
        lstm_in = torch.cat((branch_out), 1)
        #print lstm_in.size()
        lstm_out = self.lstm(lstm_in)[0]
        #lstm_out = lstm_out[0]
        lstm_out = lstm_out[:, -1, :]
        # print lstm_out.size()
        conv_in = lstm_in.permute(0, 2, 1).unsqueeze(3)  # 10, 2048, 7, 1
        # print conv_in.size()
        conv_out = self.conv(conv_in)  # 10, 64, 5, 1
        # print conv_out.size()
        conv_out = conv_out.view(x.size()[0], -1)  # 10, 320
        # print conv_out.size()

        fc_in = torch.cat([lstm_out, conv_out], 1)  # 10, 448
        # print fc_in.size()
        result = self.fc(fc_in)
        # print result.size()

        return result


if __name__ == '__main__':
    aa = torch.autograd.Variable(torch.ones(10, 7, 3, 32, 32), requires_grad=True)
    model = CNN_LSTM()
    result = model(aa)
    print result
