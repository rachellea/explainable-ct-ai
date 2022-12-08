#custom_models_selfsupervised.py
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

class BodySelfSupervised(nn.Module):
    #identical to the Body_Avg model except predicts additional 6 outputs
    #indicating what kind of data augmentation was performed.
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]
    Self-supervised learning in which the model predicts not only the n_outputs
    diseases but also predicts 6 additional numbers indicating what kind of
    data augmentation was performed (flips and rotations)"""
    def __init__(self, n_outputs):
        super(BodySelfSupervised, self).__init__()
        n_outputs = n_outputs+6 #predict the additional 6 outputs for what kind
        #of data augmentation was performed
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        #dimensions here really should be 83+6 since we have additional 6 outputs
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        x = self.fc(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class BodySelfSupervisedBranch(nn.Module):
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]
    Self-supervised learning in which the model predicts not only the n_outputs
    diseases but also predicts 6 additional numbers indicating what kind of
    data augmentation was performed (flips and rotations)
    The 6 additional numbers are predicted using a separate branch"""
    def __init__(self, n_outputs):
        super(BodySelfSupervisedBranch, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.fc_diseases = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.fc_augs = nn.Conv2d(16, 6, kernel_size = (6,6), stride=(6,6), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        #Predict diseases
        xd = self.avg_pool_and_reshape(self.fc_diseases(x)) #out shape [1,83]
        #Predict data augmentations
        xa = self.avg_pool_and_reshape(self.fc_augs(x)) #out shape [1,6]
        return torch.cat((xd,xa),dim=1)
    
    def avg_pool_and_reshape(self, x):
        #dimensions shown are for when x is a disease vector (i.e. 83)
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x
