#custom_models_less_resnet.py
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

#after resnet[0:1]: [slices, 64, 210, 210]
#after resnet[0:2]: [slices, 64, 210, 210]
#after resnet[0:3]: [slices, 64, 210, 210]
#after resnet[0:4]: [slices, 64, 105, 105]
#after resnet[0:5]: [slices, 64, 105, 105]
#after resnet[0:6]: [slices, 128, 53, 53]
#after resnet[0:7]: [slices, 256, 27, 27]
#after resnet[0:8]: [slices, 512, 14, 14] (if we go 0:-2 the len is 8)

class Body_Resnet6_Cll_Avg(nn.Module):
    """(1) ResNet18[0:6] (instead of [0:8])
       (2) conv_collapse [512, 14, 14]->[n_outputs, 1, 1]
       (3) avg pooling over slices"""
    def __init__(self, n_outputs):
        super(Body_Resnet6_Cll_Avg, self).__init__()
        self.slices = 15 #9 projections
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[0:6])) #[slices, 128, 53, 53]
        self.conv2d = nn.Sequential(nn.Conv2d(128, 64, kernel_size = (3,3), stride=(1,1), padding=0), #in_channels changed from 512 to 128
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, n_outputs, kernel_size = (45,45), stride=(45,45), padding=0)) #changed kernel size from (6,6) to (45,45)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x)
        x = self.conv2d(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x


class Body_Resnet6_Avg_Avg(nn.Module):
    """(1) ResNet18[0:6] (instead of [0:8])
       (2) conv [512, 14, 14]->[n_outputs, 44, 44]
       (3) avg pooling over 44 x 44
       (4) avg pooling over slices"""
    def __init__(self, n_outputs):
        super(Body_Resnet6_Avg_Avg, self).__init__()
        self.slices = 15 #9 projections
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[0:6])) #[slices, 128, 53, 53]
        #These convolutions are the same shape as conv_collapse except that
        #at the very end it doesn't have the huge kernel across everything
        #which means that the output isn't [83,1,1] and is instead [83,44,44]
        #so we do average pooling twice (like the conv preserve setup)
        self.conv2d = nn.Sequential(nn.Conv2d(128, 64, kernel_size = (3,3), stride=(1,1), padding=0), #in_channels changed from 512 to 128
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, n_outputs, kernel_size = (3,1), stride=(3,1), padding=0)) #changed kernel size and stride (45,45) to (3,1)
        self.avgpool_2d = nn.AvgPool2d(kernel_size=(44,44))
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
    
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x)
        x = self.conv2d(x) #out shape [slices,83,44,44]
        x_pooled = self.avgpool_2d(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x_pooled) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x


class Body_Resnet7_Cll_Avg(nn.Module):
    """(1) ResNet18[0:7] (instead of [0:8])
       (2) conv_collapse [512, 14, 14]->[n_outputs, 1, 1]
       (3) avg pooling over slices"""
    def __init__(self, n_outputs):
        super(Body_Resnet7_Cll_Avg, self).__init__()
        self.slices = 15 #9 projections
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[0:7])) #[slices, 256, 27, 27]
        self.conv2d = nn.Sequential(nn.Conv2d(256, 64, kernel_size = (3,3), stride=(1,1), padding=0), #in_channels changed from 512 to 256
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, n_outputs, kernel_size = (19,19), stride=(19,19), padding=0)) #changed kernel size from (6,6) to (19,19)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x)
        x = self.conv2d(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x


class Body_Resnet7_Avg_Avg(nn.Module):
    """(1) ResNet18[0:6] (instead of [0:8])
       (2) conv [512, 14, 14]->[n_outputs, 18, 18]
       (3) avg pooling over 18 x 18
       (4) avg pooling over slices"""
    def __init__(self, n_outputs):
        super(Body_Resnet7_Avg_Avg, self).__init__()
        self.slices = 15 #9 projections
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[0:7])) #[slices, 256, 27, 27]
        #These convolutions are the same shape as conv_collapse except that
        #at the very end it doesn't have the huge kernel across everything
        #which means that the output isn't [83,1,1] and is instead [83,18,18]
        #so we do average pooling twice (like the conv preserve setup)
        self.conv2d = nn.Sequential(nn.Conv2d(256, 64, kernel_size = (3,3), stride=(1,1), padding=0), #in_channels changed from 512 to 256
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, n_outputs, kernel_size = (3,1), stride=(3,1), padding=0)) #changed kernel size and stride (19,19) to (3,1)
        self.avgpool_2d = nn.AvgPool2d(kernel_size=(18,18))
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
    
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x)
        x = self.conv2d(x) #out shape [slices,83,18,18]
        x_pooled = self.avgpool_2d(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x_pooled) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x