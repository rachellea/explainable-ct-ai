#custom_models_misc_base.py
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

#Miscellaneous baseline models that I tried but which will not be included
#in a paper:

###############################################
# Baseline Models that End in Average Pooling #---------------------------------
###############################################
class Body_Cll_Sigmoid_Avg(nn.Module):
    """(1) ResNet18
       (2) conv_collapse [512, 14, 14]->[n_outputs, 1, 1]
       (3) apply a sigmoid to the slice predictions to force them to look like
           probabilities. This prevents the model from being able to allow
           one slice to "dominate" the final prediction. (I expect that this
           will result in worse performance since we WANT one slice to be
           able to domainte.)
       (3) avg pooling over slices"""
    def __init__(self, n_outputs):
        super(Body_Cll_Sigmoid_Avg, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.conv2d = cts.conv_collapse(n_outputs)
        self.sigmoid = nn.Sigmoid()
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = self.sigmoid(x) #apply sigmoid to make entries look like probabilities
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class Body_FC_Avg(nn.Module):
    """(1) ResNet18
       (2) FC layer from [512*14*14]->[n_outputs]. This assumes that further
           convolutions are unnecessary, i.e. it assumes that the ResNet has
           already extracted any features that are needed for disease
           classification.
       (3) avg pooling over slices"""
    def __init__(self, n_outputs):
        super(Body_FC_Avg, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.fc = nn.Sequential(nn.Linear(512*14*14, n_outputs))
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = torch.flatten(x,start_dim=1) #out shape [slices,512*14*14]
        x = self.fc(x) #out shape [slices,83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class Body_Avg_FC_Avg(nn.Module):
    """(1) ResNet18
       (2) avg pooling over 14 x 14 (as is done in ResNet)
       (3) FC layer to go from 512 to n_outputs, per slice
       (4) avg pooling over slices"""
    def __init__(self, n_outputs):
        super(Body_Avg_FC_Avg, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.avgpool_2d = nn.AvgPool2d(kernel_size=(14,14))
        self.fc = nn.Sequential(nn.Linear(512, n_outputs))
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices).to('cuda:0')
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x_pooled = self.avgpool_2d(x) #out shape [slices,512,1,1]
        x = torch.squeeze(x_pooled) #out shape [slices, 512]
        x = self.fc(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class Body_Psv_Avg_Avg(nn.Module):
    """(1) ResNet18
       (2) conv_preserve [512, 14, 14]->[n_outputs, 14, 14]
       (3) avg pooling over 14 x 14
       (4) avg pooling over slices"""
    def __init__(self, n_outputs):
        super(Body_Psv_Avg_Avg, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.conv2d = cts.conv_preserve(n_outputs)
        self.avgpool_2d = nn.AvgPool2d(kernel_size=(14,14))
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices,83,14,14]
        x_pooled = self.avgpool_2d(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x_pooled) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class Body_Psv_FC_Avg(nn.Module):
    """(1) ResNet18
       (2) conv_preserve [512, 14, 14]->[n_outputs, 14, 14]
       (3) fc layer to [n_outputs]
       (4) avg pooling over slices"""
    def __init__(self, n_outputs):
        super(Body_Psv_FC_Avg, self).__init__()
        self.slices = 15 #9 projections
        self.n_outputs = n_outputs
        self.features = cts.resnet_features()
        self.conv2d = cts.conv_preserve(n_outputs)
        self.fc = nn.Sequential(nn.Linear(n_outputs*14*14, n_outputs))        
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices,83,14,14]
        x = x.view(self.slices,self.n_outputs*14*14)
        x = self.fc(x) #out shape [slices,83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

################################################################
# Baseline Models that End in FC Layers (Destroys Slice Preds) #----------------
################################################################
class Body_Cll_FC(nn.Module):
    """(1) ResNet18
       (2) conv_collapse [512, 14, 14]->[n_outputs, 1, 1]
       (3) FC layer to go from 15*n_outputs -> n_outputs"""
    def __init__(self, n_outputs):
        super(Body_Cll_FC, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.conv2d = cts.conv_collapse(n_outputs)
        self.fc = nn.Sequential(nn.Linear(self.slices*n_outputs, n_outputs))
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices,83,1,1]
        x = torch.flatten(x) #out shape [slices*83]
        x = x.unsqueeze(0) #out shape [1, slices*83]
        x = self.fc(x) #out shape [1, 83]
        return x

class Body_Avg_FC_FC(nn.Module):
    """(1) ResNet18
       (2) avg pooling over 14 x 14 (as is done in ResNet)
       (3) FC layer to go from 512 to n_outputs, per slice
       (4) FC layer to go from 15 slices * n_outputs to n_outputs overall"""
    def __init__(self, n_outputs):
        super(Body_Avg_FC_FC, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.avgpool_2d = nn.AvgPool2d(kernel_size=(14,14))
        self.fc1 = nn.Sequential(nn.Linear(512, n_outputs),nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(self.slices*n_outputs, n_outputs))
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x_pooled = self.avgpool_2d(x) #out shape [slices,512,1,1]
        x = torch.squeeze(x_pooled) #out shape [slices, 512]
        x = self.fc1(x) #out shape [slices, 83]
        x = torch.flatten(x) #out shape [slices*83]
        x = x.unsqueeze(0) #out shape [1,slices*83]
        x = self.fc2(x) #out shape [1,83]
        return x

class Body_FC_FC(nn.Module):
    """(1) ResNet18
       (2) FC layer to go from 512*14*14 to n_outputs, per slice
       (3) FC layer to go from 15 slices * n_outputs to n_outputs overall"""
    def __init__(self, n_outputs):
        super(Body_FC_FC, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.fc1 = nn.Sequential(nn.Linear(512*14*14, n_outputs),nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(self.slices*n_outputs, n_outputs))
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = torch.flatten(x,start_dim=1) #out shape [slices,512*14*14]
        x = self.fc1(x) #out shape [slices, 83]
        x = torch.flatten(x) #out shape [slices*83]
        x = x.unsqueeze(0) #out shape [1,slices*83]
        x = self.fc2(x) #out shape [1,83]
        return x

#############################################
# Baseline Models David Dov Suggested (MIL) #-----------------------------------
#############################################
class AvgPoolMIL(nn.Module):
    """(1) ResNet18	[slices, 512, 14, 14]
       (2) Rearrange tensor to [slices*14*14, 512]
       (3) MLP to [slices*14*14, 83]
       (4) average pooling to [83]"""
    def __init__(self, n_outputs):
        super(AvgPoolMIL, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.mlp = nn.Sequential(nn.Linear(512, 256),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(256,128),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(128,n_outputs))
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices*14*14)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = x.transpose(0,1) #out shape [512,slices,14,14]
        x = x.flatten(start_dim=1) #out shape [512, slices*14*14]
        x = x.transpose(0,1) #out shape [slices*14*14, 512]
        x = self.mlp(x) #out shape [slices*14*14, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1,83,slices*14*14]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class DeepSetsMIL(nn.Module):
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) Rearrange tensor to [slices*14*14, 512]
       (3) MLP to [slices*14*14, 512]
       (4) Attention (content-based weights of size slices*14*14) to [slices*14*14, 512]
       (5) average pooling to [512]
       (6) MLP to [83]"""
    def __init__(self, n_outputs):
        super(DeepSetsMIL, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.mlp1 = nn.Sequential(nn.Linear(512, 256),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(256,256),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(256,512))
        self.mlp_attn = nn.Sequential(nn.Linear(512, 256),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(256,128),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(128,1))
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices*14*14)
        self.mlp2 = nn.Sequential(nn.Linear(512, 256),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(256,128),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(128,n_outputs))
    
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        
        #Calculate output of MLP of step (3)
        x = x.transpose(0,1) #out shape [512,slices,14,14]
        x = x.flatten(start_dim=1) #out shape [512, slices*14*14]
        x_in = x.transpose(0,1) #out shape [slices*14*14, 512]
        x1 = self.mlp1(x_in) #out shape [slices*14*14, 512]
        
        #Calculate attention weights needed in step (4)
        attn = self.mlp_attn(x_in) #out shape [slices*14*14, 1]
        
        #Apply the attention
        x2 = torch.mul(x1, attn) #out shape [slices*14*14, 512]
        
        #Step (5), Average pooling to [512]
        x2 = x2.transpose(0,1).unsqueeze(0) #out shape [1, 512, slices*14*14]
        x2 = self.avgpool_1d(x2) #out shape [1, 512, 1]
        x2 = x2.squeeze() #out shape [512]
        x2 = x2.unsqueeze(0)
        
        #Step (6), MLP to [83]
        xfinal = self.mlp2(x2) #out shape [1, 83]
        return xfinal

 