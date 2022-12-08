#custom_models_ablation.py
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

#Max pooling instead of average pooling as the final step
class Ablate_AxialNet_FinalMaxPool(nn.Module):
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) DIFFERENT: MAX pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, slices):
        super(Ablate_AxialNet_FinalMaxPool, self).__init__()
        self.slices = slices #equal to 15 for 9 projections
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.maxpool_1d = nn.MaxPool1d(kernel_size=self.slices) #DIFFERENT
        
    def forward(self, x):
        assert list(x.shape)==[1,self.slices,3,420,420]
        x = x.squeeze() #out shape [slices,3,420,420]
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        x = self.fc(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.maxpool_1d(x) #out shape [1, 83, 1] #DIFFERENT
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

#No custom convolutional layers after the ResNet18
class Ablate_AxialNet_NoCustomConv(nn.Module):
    """(1) ResNet18 [slices, 512, 14, 14]
       DIFFERENT: conv_final deleted
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, slices):
        super(Ablate_AxialNet_NoCustomConv, self).__init__()
        self.slices = slices #equal to 15 for 9 projections
        self.features = cts.resnet_features()
        #DIFFERENT: deleted custom conv layers
        self.fc = nn.Conv2d(512, n_outputs, kernel_size = (14,14), stride=(14,14), padding=0) #DIFFERENT
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        assert list(x.shape)==[1,self.slices,3,420,420]
        x = x.squeeze() #out shape [slices,3,420,420]
        x = self.features(x) #out shape [slices,512,14,14]
        #DIFFERENT: deleted custom conv layers
        x = self.fc(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

#Apply a sigmoid function after the custom conv layers so that the per-slice
#scores become per-slice probabilities
class Ablate_AxialNet_SigmoidAfterCustomConv(nn.Module):
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       DIFFERENT: Apply a sigmoid so that the per-slice scores become
       per-slice probabilities
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, slices):
        super(Ablate_AxialNet_SigmoidAfterCustomConv, self).__init__()
        self.slices = slices #equal to 15 for 9 projections
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.sigmoid = nn.Sigmoid() #DIFFERENT
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        assert list(x.shape)==[1,self.slices,3,420,420]
        x = x.squeeze() #out shape [slices,3,420,420]
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        x = self.fc(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = self.sigmoid(x) #DIFFERENT: apply a sigmoid to get 'probabilities'
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

#Use a randomly-initialized ResNet instead of a pretrained ResNet
class Ablate_AxialNet_RandomInitResNet(nn.Module): 
    """(1) ResNet18 [slices, 512, 14, 14]
           DIFFERENT: randomly initialized instead of pretrained on ImageNet
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, slices):
        super(Ablate_AxialNet_RandomInitResNet, self).__init__()
        self.slices = slices #equal to 15 for 9 projections
        
        resnet = models.resnet18(pretrained=False) #DIFFERENT - not pretrained!
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
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
