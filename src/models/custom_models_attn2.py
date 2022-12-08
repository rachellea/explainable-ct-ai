#custom_models_attn2.py
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

class Body_Cll_Avg_HAttnviaSE(nn.Module):
    """(1) ResNet18
       (2) height attention using a 3-dimensional SE Block
       (3) conv_collapse [512, 14, 14]->[n_outputs, 1, 1]
       (4) avg pooling over slices"""
    def __init__(self, n_outputs):
        super(Body_Cll_Avg_HAttnviaSE, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.SE_height = cts.SEBlock3D(channels=15, height=512, width=14, depth=14)
        self.conv2d = cts.conv_collapse(n_outputs)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        
        #Height Attention via 3D SE Block
        x = x.unsqueeze(0) #out shape [1, 45, 512, 14, 14]
        x = self.SE_height(x)
        x = torch.squeeze(x, dim=0) #out shape [45, 512, 14, 14]
        
        #Convolutions and Average Pooling
        x = self.conv2d(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class Body_Cll_Avg_FAttnviaSE(nn.Module):
    """(1) ResNet18
       (2) feature attention using a 3-dimensional SE Block
       (3) conv_collapse [512, 14, 14]->[n_outputs, 1, 1]
       (4) avg pooling over slices"""
    def __init__(self, n_outputs):
        super(Body_Cll_Avg_FAttnviaSE, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.SE_feature = cts.SEBlock3D(channels=512, height=15, width=14, depth=14)
        self.conv2d = cts.conv_collapse(n_outputs)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        
        #Feature Attention via 3D SE Block
        x = x.transpose(0,1).unsqueeze(0) #out shape [1,512,slices,14,14]
        x = self.SE_feature(x)
        x = x.squeeze(0).transpose(0,1) #out shape [slices,512,14,14]
        
        #Convolutions and Average Pooling
        x = self.conv2d(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class Body_Cll_Avg_HFWeight(nn.Module):
    """(1) ResNet18
       (2) height and/or feature weighting using weights that are learned as
           part of the model and which are the same for all inputs
           (technically not attention)
       (3) conv_collapse [512, 14, 14]->[n_outputs, 1, 1]
       (4) avg pooling over slices"""
    def __init__(self, n_outputs, use_attn_features, use_attn_height):
        super(Body_Cll_Avg_HFWeight, self).__init__()
        self.slices = 15 #9 projections
        self.sigmoid = nn.Sigmoid()
        self.use_attn_features = use_attn_features
        self.use_attn_height = use_attn_height
        self.features = cts.resnet_features()
        if self.use_attn_features:
            self.attn_features = nn.Parameter(torch.ones((1,512,1,1), dtype=torch.float32),requires_grad=True)
        if self.use_attn_height:
            self.attn_height = nn.Parameter(torch.ones((self.slices,1,1,1), dtype=torch.float32),requires_grad=True)
        self.conv2d = cts.conv_collapse(n_outputs)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        
        #Height and/or Feature Attention
        if self.use_attn_features:
            af = self.sigmoid(self.attn_features)
            x = torch.mul(x, af)
        
        if self.use_attn_height:
            ah = self.sigmoid(self.attn_height)
            x = torch.mul(x, ah)
        
        #Convolutions and Average Pooling
        x = self.conv2d(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class Body_Cll_Avg_HSliceWeight(nn.Module):
    """(1) ResNet18
       (2) conv_collapse [512, 14, 14]->[n_outputs, 1, 1]
       (3) weights over [n_outputs] x [# slices] so that the model can learn
           which vertical slices are most likely to contribute to a particular
           disease
       (4) avg pooling over slices
    The cumulative effect of steps (3) and (4) is weighted average pooling"""
    def __init__(self, n_outputs):
        super(Body_Cll_Avg_HSliceWeight, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.attn_hslice = nn.Parameter(torch.ones((1, n_outputs, self.slices), dtype=torch.float32),requires_grad=True)
        self.conv2d = cts.conv_collapse(n_outputs)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        #Convolutions and Weighted Average Pooling
        x = self.conv2d(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = torch.mul(x, self.attn_hslice) #apply the weights
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class Body_Cll_Avg_HSliceAttn(nn.Module):
    """(1) ResNet18
       (2) conv_collapse [512, 14, 14]->[n_outputs, 1, 1]
       (3) learned attention weights over [n_outputs] x [# slices] so that the
           model can learn which vertical slices are most likely to contribute
           to a particular disease by considering the slice risks of all of the
           different diseases.
       (4) avg pooling over slices
    The cumulative effect of steps (3) and (4) is weighted average pooling"""
    def __init__(self, n_outputs):
        super(Body_Cll_Avg_HSliceAttn, self).__init__()
        self.slices = 15 #9 projections
        self.n_outputs = n_outputs
        self.features = cts.resnet_features()
        self.conv2d = cts.conv_collapse(n_outputs)
        self.sigmoid = nn.Sigmoid()
        self.attn_hslice_fc = nn.Sequential(nn.Linear(n_outputs*self.slices, n_outputs*self.slices))
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        #Convolutions
        x = self.conv2d(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        #HSliceAttn:
        x_flat = x.contiguous().view(1, self.n_outputs*self.slices)
        attn_flat = self.sigmoid(self.attn_hslice_fc(x_flat))
        attn_map = attn_flat.view(1,self.n_outputs,self.slices)
        x_times_attn_map = torch.mul(x, attn_map) #out shape [1, 83, slices]
        #Average Pooling
        x2 = self.avgpool_1d(x_times_attn_map) #out shape [1, 83, 1]
        x3 = torch.squeeze(x2, dim=2) #out shape [1, 83]
        return x3

class Body_Cll_Avg_SpatialAttn(nn.Module):
    """(1) ResNet18
       (2) conv_collapse [512, 14, 14]->[n_outputs, 1, 1] including spatial
           attention after the first conv layer
       (3) avg pooling over slices"""
    def __init__(self, n_outputs):
        super(Body_Cll_Avg_SpatialAttn, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.sigmoid = nn.Sigmoid()
        self.conv1, self.attn, self.conv2 = cts.conv_collapse_spatial_attn(n_outputs)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        
        #Convolutions and Spatial Attention
        x1 = self.conv1(x)
        attn = self.sigmoid(self.attn(x))
        x1_times_attn = torch.mul(x1,attn)
        x2 = self.conv2(x1_times_attn) #out shape [slices,83,1,1]
        
        #Average Pooling
        x3 = torch.squeeze(x2) #out shape [slices, 83]
        x4 = x3.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x5 = self.avgpool_1d(x4) #out shape [1, 83, 1]
        x6 = torch.squeeze(x5, dim=2) #out shape [1, 83]
        return x6

class Body_Cll_Avg_SpatialAttn2(nn.Module):
    """(1) ResNet18
       (2) conv_collapse [512, 14, 14]->[n_outputs, 1, 1] including spatial
           attention at the end of the conv layers. Attention maps have
           shape [slices,83,1,1] which in retrospect is really weird and no
           wonder this doesn't work well.
       (3) avg pooling over slices"""
    def __init__(self, n_outputs):
        super(Body_Cll_Avg_SpatialAttn2, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.sigmoid = nn.Sigmoid()
        self.conv1, self.attn, self.conv2 = cts.conv_collapse_spatial_attn2(n_outputs)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        
        #Convolutions and Spatial Attention
        x = self.conv1(x)
        x1 = self.conv2(x)
        attn = self.sigmoid(self.attn(x))
        x1_times_attn = torch.mul(x1,attn) #out shape [slices,83,1,1]
        
        #Average Pooling
        x3 = torch.squeeze(x1_times_attn) #out shape [slices, 83]
        x4 = x3.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x5 = self.avgpool_1d(x4) #out shape [1, 83, 1]
        x6 = torch.squeeze(x5, dim=2) #out shape [1, 83]
        return x6

class Body_Cll_Avg_SpatialAttn3(nn.Module):
    """(1) ResNet18
       (2) conv_collapse [512, 14, 14]->[n_outputs, 1, 1] including spatial
           attention at second to last conv layer. Attention maps have
           shape [slices,16,6,6]
       (3) avg pooling over slices"""
    def __init__(self, n_outputs):
        super(Body_Cll_Avg_SpatialAttn3, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.sigmoid = nn.Sigmoid()
        self.conv1, self.attn, self.conv2, self.conv3 = cts.conv_collapse_spatial_attn3(n_outputs)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        
        #Convolutions and Spatial Attention
        #this way of doing attention is like a little bubble in the network
        #where one branch is attention and the other is conv and the only
        #difference between the attention and the conv is that the attention
        #has a sigmoid applied to it.
        x1 = self.conv1(x)
        attn = self.sigmoid(self.attn(x1))
        c2out = self.conv2(x1)
        c2out_times_attn = torch.mul(c2out,attn)
        x2 = self.conv3(c2out_times_attn) #out shape [slices,83,1,1]
        
        #Average Pooling
        x3 = torch.squeeze(x2) #out shape [slices, 83]
        x4 = x3.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x5 = self.avgpool_1d(x4) #out shape [1, 83, 1]
        x6 = torch.squeeze(x5, dim=2) #out shape [1, 83]
        return x6