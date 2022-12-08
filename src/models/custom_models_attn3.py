#custom_models_attn3.py
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

#TODO come up with cleverer ways to learn the attention including 3D convolution
#that considers the entire volume before it decides on the slice attention.

class BodyAvg_SpatialAtt3_3(nn.Module): #6/29/2020, updated 7/16/2020 because there
    #was a stupid bug in which I applied a Softmax to a single-valued attention
    #output forcing it to always be equal to 1.
    """Spatial attention implemented in a way that more closely matches
    the Learn to Pay Attention feature. Specifically instead of having
    a different attention value for each channel, there is only one attention
    value across all channels for a particular spatial location (i.e. there is
    one attention value per spatial location). The input to calculate each
    attention value is a features vector.
       (1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, nonlinearity):
        super(BodyAvg_SpatialAtt3_3, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features() #out shape [512, 14, 14]
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0), #out shape [64, 12, 12]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0), #out shape [32, 10, 10]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=0), #out shape [16, 8, 8]
            nn.ReLU(inplace=True))
        #FC layer that takes in a feature vector and outputs an attention map
        #Note that a softmax is applied to force the different locations to
        #compete with each other
        self.attn_fc = nn.Linear(16, 1)
        if nonlinearity == 'softmax':
            #TODO double check that we want dim = 1 to make the 8x8 elements compete with each other
            self.nonlinearity = nn.Softmax(dim = 1)
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        
        #Remainder of model:
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=0), #out [16, 6, 6]
            nn.ReLU(inplace=True))
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
    
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d_1(x) #out shape [slices, 16, 8H, 8W]
        
        #Calculate attention map
        x1 = x.transpose(1,3) #out shape [slices, 8W, 8H, 16]
        x1 = x1.flatten(start_dim = 0, end_dim = 2) #out shape [slices*8W*8H, 16]
        attn_values = self.attn_fc(x1) #out shape [slices*8W*8H, 1]
        attn_values = attn_values.reshape((self.slices,64)) #out shape [slices, 64]
        attn_squished = self.nonlinearity(attn_values) #out shape [slices, 64]
        attn_map = (attn_squished.reshape((self.slices, 8, 8, 1))).transpose(1,3) #out shape [slices,1,8H,8W]
        
        #Multiply representation by attention map
        x_times_attn = torch.mul(x,attn_map)
        
        #Finish calculating the predictions
        x2 = self.conv2d_2(x_times_attn) #out shape [slices, 16, 6, 6]
        x2 = self.fc(x2) #out shape [slices,83,1,1]
        x2 = torch.squeeze(x2) #out shape [slices, 83]
        slice_pred = x2.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x2 = self.avgpool_1d(slice_pred) #out shape [1, 83, 1]
        x2 = torch.squeeze(x2, dim=2) #out shape [1, 83]
        #return x2
        return x2, slice_pred, attn_map

class BodyAvg_SpatialAtt3_3_Spy3D(nn.Module): #7/17/2020
    """Same as BodyAvg_SpatialAtt3_3 except it 'spies' on the full height of
    the scan when learning the attention weights for each slice"""
    def __init__(self, n_outputs, nonlinearity):
        super(BodyAvg_SpatialAtt3_3_Spy3D, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features() #out shape [512, 14, 14]
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0), #out shape [64, 12, 12]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0), #out shape [32, 10, 10]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=0), #out shape [16, 8, 8]
            nn.ReLU(inplace=True))
        #FC layer that takes in a feature vector and outputs an attention map
        #Note that a softmax is applied to force the different locations to
        #compete with each other
        self.attn_fc = nn.Linear(16*self.slices, self.slices)
        if nonlinearity == 'softmax':
            #NOTE: this is dim = 0 here because we apply it to attn_values
            #with shape [8H*8W, slices] and we want to constrain for each slice
            #that the 8H*8W elements sum to one (and must compete against each other)
            self.nonlinearity = nn.Softmax(dim = 0)
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        
        #Remainder of model:
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=0), #out [16, 6, 6]
            nn.ReLU(inplace=True))
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
    
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d_1(x) #out shape [slices, 16, 8H, 8W]
        
        #Calculate attention map
        x1 = x.flatten(start_dim = 2, end_dim = 3) #out shape [slices, 16, 8H*8W]
        x1 = x1.flatten(start_dim = 0, end_dim = 1) #out shape [slices*16, 8H*8W]
        x1 = x1.transpose(0,1) #out shape [8H*8W, slices*16]
        #for each spatial position within the 8x8, get a feature vector across
        #the whole height, which is slices*16, and use this to determine the
        #attention element at that spatial position for each of the slices
        attn_values = self.attn_fc(x1) #out shape [8H*8W, slices]
        attn_squished = self.nonlinearity(attn_values) #out shape [8H*8W, slices]
        attn_map = attn_squished.transpose(0,1) #out shape [slices, 8H*8W]
        attn_map = attn_map.reshape((self.slices, 1, 8, 8)) #out shape [slices,1,8H,8W]
        
        #Multiply representation by attention map
        x_times_attn = torch.mul(x,attn_map)
        
        #Finish calculating the predictions
        x2 = self.conv2d_2(x_times_attn) #out shape [slices, 16, 6, 6]
        x2 = self.fc(x2) #out shape [slices,83,1,1]
        x2 = torch.squeeze(x2) #out shape [slices, 83]
        slice_pred = x2.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x2 = self.avgpool_1d(slice_pred) #out shape [1, 83, 1]
        x2 = torch.squeeze(x2, dim=2) #out shape [1, 83]
        return x2
        #return x2, slice_pred, attn_map

class BodyAvg_SpatialAtt3_4(nn.Module): #7/17/2020
    """Same as BodyAvg_SpatialAtt3_3 except that the attention happens one
    conv layer later, over the 6x6 representation instead of over the 8x8
    representation"""
    def __init__(self, n_outputs, nonlinearity):
        super(BodyAvg_SpatialAtt3_4, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features() #out shape [512, 14, 14]
        self.conv2d = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0), #out shape [64, 12, 12]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0), #out shape [32, 10, 10]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=0), #out shape [16, 8, 8]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=0), #out [16, 6, 6]
            nn.ReLU(inplace=True))
        #FC layer that takes in a feature vector and outputs an attention map
        #Note that a softmax is applied to force the different locations to
        #compete with each other
        self.attn_fc = nn.Linear(16, 1)
        if nonlinearity == 'softmax':
            self.nonlinearity = nn.Softmax(dim = 1)
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
    
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6H, 6W]
        
        #Calculate attention map
        x1 = x.transpose(1,3) #out shape [slices, 6W, 6H, 16]
        x1 = x1.flatten(start_dim = 0, end_dim = 2) #out shape [slices*6W*6H, 16]
        attn_values = self.attn_fc(x1) #out shape [slices*6W*6H, 1]
        attn_values = attn_values.reshape((self.slices,36)) #out shape [slices, 64]
        attn_squished = self.nonlinearity(attn_values) #out shape [slices, 64]
        attn_map = (attn_squished.reshape((self.slices, 6, 6, 1))).transpose(1,3) #out shape [slices,1,6H,6W]
        
        #Multiply representation by attention map
        x_times_attn = torch.mul(x,attn_map)
        
        #Finish calculating the predictions
        x2 = self.fc(x_times_attn) #out shape [slices,83,1,1]
        x2 = torch.squeeze(x2) #out shape [slices, 83]
        slice_pred = x2.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x2 = self.avgpool_1d(slice_pred) #out shape [1, 83, 1]
        x2 = torch.squeeze(x2, dim=2) #out shape [1, 83]
        #return x2
        return x2, slice_pred, attn_map
