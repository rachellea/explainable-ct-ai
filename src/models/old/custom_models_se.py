#custom_models_se.py
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

#Last hope for helping memory issues (should also help speed)
#https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class Body_Fully2D_83Avg_HFAttn_SE(nn.Module):
    """Model for big data.
    Fully 2D convolutional on slices. Get 83 probabilities for each slice
    and average them together to get the final prediction
    Assumes inputs are proj9.
    Includes Height Feature Attention.
    Includes SE Blocks in the 2D Reducing Convs."""
    def __init__(self, n_outputs):
        super(Body_Fully2D_83Avg_HFAttn_SE, self).__init__()      
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #attention
        self.sigmoid = nn.Sigmoid()
        means_features = torch.Tensor(np.full((1,512,1,1),1.0,dtype='float'))
        stds_features = torch.Tensor(np.full((1,512,1,1),0.05,dtype='float'))
        self.attn_features = nn.Parameter(torch.normal(mean=means_features,std=stds_features),requires_grad=True)
        
        means_height = torch.Tensor(np.full((15,1,1,1),1.0,dtype='float'))
        stds_height = torch.Tensor(np.full((15,1,1,1),0.05,dtype='float'))
        self.attn_height = nn.Parameter(torch.normal(mean=means_height,std=stds_height),requires_grad=True)
        
        #conv2d input [134,512,14,14]
        self.conv2d_reducing = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            SEBlock2D(channels=64, height=12, width=12),
            
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            SEBlock2D(channels=32, height=10, width=10),
            
            nn.Conv2d(32, 16, kernel_size = (2,2), stride=(2,2), padding=0),
            nn.ReLU(inplace=True),
            SEBlock2D(channels=16, height=5, width=5),
            
            nn.Conv2d(16, 16, kernel_size = (2,2), stride=(2,2), padding=0), #TODO!!!: THINK ABOUT THIS STEP: MAY BE CUTTING OFF 1/5 OF INFO B/C PADDING=0
            nn.ReLU(inplace=True),
            SEBlock2D(channels=16, height=2, width=2),
            
            nn.Conv2d(16, n_outputs, kernel_size = (2,2), stride=(2,2), padding=0))
        
        self.avgpool = nn.AvgPool1d(kernel_size=15)
        #TODO try with other kinds of pooling
        
    def forward(self, x):
        shape = list(x.size())
        batch_size = int(shape[0])
        assert batch_size == 1
        x = x.view(batch_size*15,3,420,420)
        x = self.features(x) # out shape [134,512,14,14]
        
        af = self.sigmoid(self.attn_features)
        ah = self.sigmoid(self.attn_height)
        x = torch.mul(x, af)
        x = torch.mul(x, ah)
        
        x = self.conv2d_reducing(x) #out shape [134,83,1,1]
        x = torch.squeeze(x) #out shape [134, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, 134]
        x = self.avgpool(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

################################################################################
class Fully2D_HAttnviaSE(nn.Module):
    """Model for big data.
    Fully 2D convolutional on slices. Get 83 probabilities for each slice
    and average them together to get the final prediction
    Assumes inputs are proj9.
    Includes Height Attention which is learned through a 3D SE block"""
    def __init__(self, n_outputs):
        super(Fully2D_HAttnviaSE, self).__init__()      
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        self.SE_height = SEBlock3D(channels=15, height=512, width=14, depth=14)
        
        #conv2d input [134,512,14,14]
        self.conv2d_reducing = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (2,2), stride=(2,2), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (2,2), stride=(2,2), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, n_outputs, kernel_size = (2,2), stride=(2,2), padding=0))
        
        self.avgpool = nn.AvgPool1d(kernel_size=15)
        #TODO try with other kinds of pooling
        
    def forward(self, x):
        shape = list(x.size())
        batch_size = int(shape[0])
        assert batch_size == 1
        x = x.view(batch_size*15,3,420,420)
        x = self.features(x) # out shape [45,512,14,14]
        
        #Height attention learned through 3D SE Block
        x = x.unsqueeze(0) #shape [1, 45, 512, 14, 14]
        x = self.SE_height(x)
        x = torch.squeeze(x, dim=0) #shape [45, 512, 14, 14]
        
        #Convs and avg pool
        x = self.conv2d_reducing(x) #out shape [45,83,1,1]
        x = torch.squeeze(x) #out shape [45, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, 45]
        x = self.avgpool(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

################################################################################

class SEBlock2D(nn.Module):
    def __init__(self, channels, height, width):
        """References:
        https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
        https://github.com/ai-med/squeeze_and_excitation/blob/master/squeeze_and_excitation/squeeze_and_excitation.py
        https://github.com/jonnedtc/Squeeze-Excitation-PyTorch/blob/master/networks.py
        """
        super(SEBlock2D, self).__init__()
        reduction = 16        
        #torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False,
        #count_include_pad=True, divisor_override=None)
        self.squeeze_operation = nn.AvgPool2d((height, width))
        
        self.excitation_operation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        #calculate channel descriptor z with global average pooling
        z = self.squeeze_operation(x) #for input x of shape [134, 64, 105, 105] this produces output of [134, 64, 1, 1]
        z = z.squeeze() #e.g. out [134, 64]
        
        #calculate activations s with two feedforward layers
        s = self.excitation_operation(z) #e.g. out [134, 64], where batch size is 134
        s_tile = s.unsqueeze(dim=2).unsqueeze(dim=3) #e.g. out [134, 64, 1, 1]
        
        #Final step: re-weight the channels of the input with the activations s
        #TODO make sure this is correct (double check the broadcasting)
        out = torch.mul(x, s_tile)
        return out

class SEBlock3D(nn.Module):
    def __init__(self, channels, height, width, depth):
        super(SEBlock3D, self).__init__()
        reduction = 4 #use a much lower reduction ratio because far fewer channels   
        self.squeeze_operation = nn.AvgPool3d((height, width, depth))
        
        self.excitation_operation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        #calculate channel descriptor z with global average pooling
        z = self.squeeze_operation(x)
        z = z.squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)
        
        #calculate activations s with two feedforward layers
        s = self.excitation_operation(z)
        s_tile = s.unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4)
        
        #Final step: re-weight the channels of the input with the activations s
        #TODO make sure this is correct (double check the broadcasting)
        out = torch.mul(x, s_tile)
        return out
