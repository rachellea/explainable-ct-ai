#components.py
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

#############
# Functions #-------------------------------------------------------------------
#############
def reshape_x(x, slices):
    shape = list(x.size())
    batch_size = int(shape[0])
    assert batch_size == 1
    return x.view(batch_size*slices,3,420,420)

def reshape_x_triple(x, slices):
    shape = list(x.size())
    batch_size = int(shape[0])
    assert batch_size == 1
    return x.view(batch_size*slices,3,420,210)

def resnet_features():
    resnet = models.resnet18(pretrained=True)
    return nn.Sequential(*(list(resnet.children())[:-2]))

def conv_collapse(n_outputs): #here for backwards compatibility with misc_base
    """Return 2d conv layers that collapse the representation as follows:
    input: [512, 14, 14]
           [64, 12, 12]
           [32, 10, 10]
           [16, 8, 8]
           [16, 6, 6]
           [n_outputs, 1, 1]"""
    return nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0))

def final_conv():
    """Return 2d conv layers that collapse the representation as follows:
    input: [512, 14, 14]
           [64, 12, 12]
           [32, 10, 10]
           [16, 8, 8]
           [16, 6, 6]"""
    return nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True))

def final_conv_triple():
    """Return 2d conv layers that collapse the representation as follows:
    input: [512, 14, 7]
           [64, 12, 5]
           [32, 10, 3]
           [16, 8, 1]"""
    return nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True))

def conv_collapse_spatial_attn(n_outputs):
    """Return 2d conv layers that collapse the representation, with a spatial
    attention branch in the middle"""
    conv1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True))
    
    attn = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0))
    
    conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0))
    return conv1, attn, conv2

def conv_collapse_spatial_attn2(n_outputs):
    """Return 2d conv layers that collapse the representation, with a spatial
    attention branch at the end"""
    conv1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True))
            
    attn = nn.Sequential(
            nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0))
            
    conv2 = nn.Sequential(
            nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0))
    return conv1, attn, conv2

def conv_collapse_spatial_attn3(n_outputs):
    """Return 2d conv layers that collapse the representation, with a spatial
    attention branch at the second to last layer"""
    conv1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True))
    
    attn = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=0))
    
    conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True))
      
    conv3 = nn.Sequential(
            nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0))
    return conv1, attn, conv2, conv3

def conv_collapse_triple_spatial_attn(n_outputs_lung, n_outputs_heart, spatial):
    """Return 2d conv layers that collapse the representation, with a spatial
    attention branch in the middle.
    For triple crop so the final convolutional layer is different for heart
    than for lungs."""
    conv1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True))
    
    if spatial:
        attn = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0))
    
    conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True))
    #Difference: deleted the last convolutional layer because not enough size
    #left on the shorter dimension
    
    conv3_lung = nn.Conv2d(16, n_outputs_lung, kernel_size = (6,1), stride=(6,1), padding=0)
    conv3_heart = nn.Conv2d(16, n_outputs_heart, kernel_size = (6,1), stride=(6,1), padding=0)
    
    if spatial:
        return conv1, attn, conv2, conv3_lung, conv3_heart
    else:
        return conv1, None, conv2, conv3_lung, conv3_heart

def conv_preserve(n_outputs):
    """Return 2d conv layers that reduce the number of features but use
    padding to preserve the 14 height x 14 width spatial dimension:
           [512, 14, 14]
           [64, 14, 14]
           [32, 14, 14]
           [16, 14, 14]
           [16, 14, 14]
           [n_outputs, 14, 14]"""
    return nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, n_outputs, kernel_size = (3,3), stride=(1,1), padding=1))

###########
# Classes #---------------------------------------------------------------------
###########
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
