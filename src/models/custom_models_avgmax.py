#custom_models_avgmax.py
#Copyright (c) 2020 Rachel Lea Ballantyne Draelos

#MIT License

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE

import copy
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms, models, utils

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

#Import components
from . import components as cts

class BodyAvgMax(nn.Module):
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs] as well as max pooling
           over slices to get [n_outputs] with a learned ratio between them
           that is disease specific.
           This allows the model to use the avg more or use the max
           more depending on the disease. The hypothesis is that the max
           will be more useful for focal diseases while the avg will be more
           useful for diffuse diseases. (However it may not have a huge effect
           since the model is allowed to scale the raw scores differently for
           different diseases which in effect can alter the avg's properties
           to be more like a max, if the score for one slice is massively
           inflated)
           (lambda)*(avg pool output) + (1 - lambda)(max pool output)"""
    def __init__(self, n_outputs):
        super(BodyAvgMax, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        #init lambda as 0.5 so that it starts off with equal weight between
        #the avg and the max:
        self.lambda_for_avg = nn.Parameter(0.5*torch.ones((1, n_outputs), dtype=torch.float32),requires_grad=True)
        print('init self.lambda_for_avg as',self.lambda_for_avg)
        self.avgpool_1d = nn.AvgPool1d(kernel_size = self.slices)
        self.maxpool_1d = nn.MaxPool1d(kernel_size = self.slices)
    
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        x = self.fc(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        
        #Avg pooling output
        x_avg = self.avgpool_1d(x) #out shape [1, 83, 1]
        x_avg = torch.squeeze(x_avg, dim=2) #out shape [1, 83]
        
        #Max pooling output
        x_max = self.maxpool_1d(x) #out shape [1, 83, 1]
        x_max = torch.squeeze(x_max, dim=2) #out shape [1, 83]
        
        #Combine them
        x_final = (self.lambda_for_avg*x_avg)+((1-self.lambda_for_avg)*x_max)
        return x_final

