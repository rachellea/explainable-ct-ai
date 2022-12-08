#custom_models_base.py
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

class AxialNet(nn.Module):
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, slices):
        super(AxialNet, self).__init__()
        self.slices = slices #equal to 15 for 9 projections
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        assert list(x.shape)==[1,self.slices,3,420,420]
        x = x.squeeze() #out shape [slices,3,420,420]
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        x = self.fc(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class AvgPoolMILFinal(nn.Module): #Baseline 2
    """(1) ResNet18	[slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) Rearrange tensor to [slices*6*6, 16]
       (4) MLP to [slices*6*6, n_outputs]
       (5) Rearrange tensor to [slices, n_outputs, 6, 6]
       (6) 2d pooling over [6, 6] -> out [slices, n_outputs, 1, 1]
       (7) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, slices):
        super(AvgPoolMILFinal, self).__init__()
        self.slices = slices
        self.n_outputs = n_outputs
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.mlp = nn.Sequential(nn.Linear(16, 64),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(64,n_outputs))
        self.avgpool_2d = nn.AvgPool2d(kernel_size=(6,6))
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x): #TODO check tensor rearrangements to ensure proper reversal
        #ResNet18 
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        
        #Rearrange tensor
        x = x.transpose(0,1) #out shape [16,slices,6,6]
        x = x.flatten(start_dim=1) #out shape [16, slices*6*6]
        x = x.transpose(0,1) #out shape [slices*6*6, 16]
        
        #MLP
        x = self.mlp(x) #out shape [slices*6*6, 83]
        
        #Rearrange tensor
        x = x.transpose(0,1) #out shape [83, slices*6*6]
        x = x.reshape((self.n_outputs, self.slices, 6, 6)) #out shape [83, slices, 6, 6]
        x = x.transpose(0,1) #out shape [slices, 83, 6, 6]
        
        #2D pooling
        x = self.avgpool_2d(x) #out shape [slices, 83, 1, 1]
        
        #Rearrange tensor, 1D pooling (same as AxialNet final steps)
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class DeepSetsMILFinal(nn.Module): #Baseline 3
    """(1) ResNet [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) Rearrange tensor to [slices*6*6, 16]
       (4) MLP to [slices*6*6, 16], i.e. go from 16 to a different 16
       (5) Rearrange again to [slices, 16, 6, 6]
       (6) 2d pooling over [6, 6] -> out [slices, 16, 1, 1]
       (7) MLP to [slices, n_outputs]
       (8) Avg pooling over slices to get [n_outputs]
    
    Note: David strongly suggests that I use this model as a baseline. However
    after reading the paper I am convinced it is a bad baseline because it
    explicitely assumes that the ordering of the elements doesn't matter, but
    in a CT scan the ordering in all directions matters.
    
    https://www.inference.vc/deepsets-modeling-permutation-invariance/
    The high-level description of the full architecture is now reasonably
    straightforward - transform your inputs into some latent space, destroy
    the ordering information in the latent space by applying the sum, and
    then transform from the latent space to the final output."""
    def __init__(self, n_outputs, slices):
        super(DeepSetsMILFinal, self).__init__()
        self.slices = slices
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.mlp1 = nn.Sequential(nn.Linear(16, 16),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(16,16),
                                 nn.ReLU(inplace=True))
        self.avgpool_2d = nn.AvgPool2d(kernel_size=(6,6))
        self.mlp2 = nn.Sequential(nn.Linear(16, 64),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(64,n_outputs))
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
    
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        
        #***Transform your inputs into some latent space***
        #Here we are considering the output of self.conv2d to be the inputs.
        #Calculate output of MLP of step (4)
        x = x.transpose(0,1) #out shape [16,slices,6,6]
        x = x.flatten(start_dim=1) #out shape [16, slices*6*6]
        x = x.transpose(0,1) #out shape [slices*6*6, 16]
        x = self.mlp1(x) #out shape [slices*6*6, 16]
        
        #Rearrange tensor
        x = x.transpose(0,1) #out shape [16, slices*6*6]
        x = x.reshape((16, self.slices, 6, 6)) #out shape [16, slices, 6, 6]
        x = x.transpose(0,1) #out shape [slices, 16, 6, 6]
        
        #***Destroy the ordering information in the latent space by applying
        #the sum***
        #Step (6), Average pooling to [16]
        x = self.avgpool_2d(x) #out shape [slices, 16, 1, 1]
        x = torch.squeeze(x) #out shape [slices, 16]
        
        #***Transform from the latent space to the final output***
        x = self.mlp2(x) #out shape [slices, 83]
        
        #Step (8) Average pooling over slices
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class BodyCAM(nn.Module):
    """Class Activation Mapping (CAM) Baseline
       (1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) Average pooling over the feature maps (i.e. over [slices, 6, 6]) to
           produce 16 values, one for each feature
       (4) a single FC layer to [n_outpts]"""
    def __init__(self, n_outputs, slices):
        super(BodyCAM, self).__init__()
        self.slices = slices #equal to 15 for 9 projections
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.avgpool_3d = nn.AvgPool3d(kernel_size=(self.slices,6,6))
        self.fc = nn.Linear(16, n_outputs)
        
    def forward(self, x):
        assert list(x.shape)==[1,self.slices,3,420,420]
        x = x.squeeze() #out shape [slices,3,420,420]
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        #rearrange to [N, C, D, H, W]; kernel size is for (D, H, W)
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 16, slices, 6, 6]
        #global average pooling over all dimensions except the feature dim
        #(the activation maps are 3D, with size [D x H x W])
        x = self.avgpool_3d(x) #out shape [1, 16, 1, 1, 1]
        #flatten
        x = x.squeeze(2).squeeze(2).squeeze(2) #out shape [1, 16]
        #fc
        x = self.fc(x) #out shape [1,n_outputs]
        return x