#custom_models_triplecrop.py
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

class BodyAvgTripleSharedConvSharedLung(nn.Module):
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) shared conv_final to [slices, 16, 8, 1]
       (3) FC layers (implemented via conv) to [n_outputs, 1, 1]
           There is a separate FC layer for the heart, but the two lungs
           SHARE an FC layer.
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs_lung, n_outputs_heart, slices):
        super(BodyAvgTripleSharedConvSharedLung, self).__init__()
        self.slices = slices #e.g. 15 for 9 projections
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv_triple()
        self.fc_heart = nn.Conv2d(16, n_outputs_heart, kernel_size = (8,1), stride=(8,1), padding=0)
        self.fc_lung = nn.Conv2d(16, n_outputs_lung, kernel_size = (8,1), stride=(8,1), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
    
    def extract_organ_representation(self, x, fc_layer):
        assert list(x.shape)==[1,self.slices,3,420,210]
        x = x.squeeze() #out shape [slices,3,420,210]
        x = self.features(x) #out shape [slices,512,14,7]
        x = self.conv2d(x) #out shape [slices, 16, 8, 1]
        x = fc_layer(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x
    
    def forward(self, x):
        heart = self.extract_organ_representation(x['heart'],self.fc_heart)
        left_lung = self.extract_organ_representation(x['left_lung'],self.fc_lung)
        right_lung = self.extract_organ_representation(x['right_lung'],self.fc_lung)
        return torch.cat((heart,left_lung,right_lung),1)

class BodyAvgTripleSharedConv(nn.Module):
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) shared conv_final to [slices, 16, 8, 1]
       (3) FC layers (implemented via conv) to [n_outputs, 1, 1]
           There is a separate FC layer for each organ: right lung, heart,
           left lung
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs_lung, n_outputs_heart):
        super(BodyAvgTripleSharedConv, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv_triple()
        self.fc_heart = nn.Conv2d(16, n_outputs_heart, kernel_size = (8,1), stride=(8,1), padding=0)
        self.fc_left_lung = nn.Conv2d(16, n_outputs_lung, kernel_size = (8,1), stride=(8,1), padding=0)
        self.fc_right_lung = nn.Conv2d(16, n_outputs_lung, kernel_size = (8,1), stride=(8,1), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
    
    def extract_organ_representation(self, x, fc_layer):
        x = cts.reshape_x_triple(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 8, 1]
        x = fc_layer(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x
    
    def forward(self, x):
        heart = self.extract_organ_representation(x['heart'],self.fc_heart)
        left_lung = self.extract_organ_representation(x['left_lung'],self.fc_left_lung)
        right_lung = self.extract_organ_representation(x['right_lung'],self.fc_right_lung)
        return torch.cat((heart,left_lung,right_lung),1)

class BodyAvgTripleSeparateConv(nn.Module):
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 8, 1]
           There is a separate conv_final for each organ: right lung, heart,
           left lung
       (3) FC layers (implemented via conv) to [n_outputs, 1, 1]
           There is a separate FC layer for each organ: right lung, heart,
           left lung
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs_lung, n_outputs_heart):
        super(BodyAvgTripleSeparateConv, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.conv2d_heart = cts.final_conv_triple()
        self.conv2d_left_lung = cts.final_conv_triple()
        self.conv2d_right_lung = cts.final_conv_triple()
        self.fc_heart = nn.Conv2d(16, n_outputs_heart, kernel_size = (8,1), stride=(8,1), padding=0)
        self.fc_left_lung = nn.Conv2d(16, n_outputs_lung, kernel_size = (8,1), stride=(8,1), padding=0)
        self.fc_right_lung = nn.Conv2d(16, n_outputs_lung, kernel_size = (8,1), stride=(8,1), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
    
    def extract_organ_representation(self, x, conv_layers, fc_layer):
        x = cts.reshape_x_triple(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = conv_layers(x) #out shape [slices, 16, 8, 1]
        x = fc_layer(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x
    
    def forward(self, x):
        heart = self.extract_organ_representation(x['heart'], self.conv2d_heart, self.fc_heart)
        left_lung = self.extract_organ_representation(x['left_lung'], self.conv2d_left_lung, self.fc_left_lung)
        right_lung = self.extract_organ_representation(x['right_lung'], self.conv2d_right_lung, self.fc_right_lung)
        return torch.cat((heart,left_lung,right_lung),1)

##############
# DEPRECATED #------------------------------------------------------------------
##############
class Body_Cll_Avg_Triple_SpatialAttn(nn.Module):
    """Triple crops model.
       (1) ResNet18
       (2) conv_collapse [512, 14, 14]->[n_outputs, 1, 1] including spatial
           attention after the first conv layer if <spatial> is True
       (3) avg pooling over slices"""
    def __init__(self, n_outputs_lung, n_outputs_heart, spatial):
        super(Body_Cll_Avg_Triple_SpatialAttn, self).__init__()
        self.slices = 15 #9 projections
        self.spatial = spatial
        self.features = cts.resnet_features()
        self.sigmoid = nn.Sigmoid()
        self.conv1, self.attn, self.conv2, self.conv3_lung, self.conv3_heart = cts.conv_collapse_triple_spatial_attn(n_outputs_lung, n_outputs_heart, spatial)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
    
    def extract_organ_representation(self, x, convfinal):
        """x is a crop from the full volume representing the right_lung, heart,
        or left_lung"""
        x = cts.reshape_x_triple(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,7]
        
        if self.spatial:
            #Convolutions and Spatial Attention
            x1 = self.conv1(x)
            attn = self.sigmoid(self.attn(x))
            x1_times_attn = torch.mul(x1,attn)
            x2 = self.conv2(x1_times_attn)
        else:
            #Convolutions Only
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            attn = None #placeholder for return
        
        #Organ-Specific Final Convolutional Layer
        x2_organ = convfinal(x2) #out shape [slices,n_outputs,1,1]
        
        #Average Pooling
        x3 = torch.squeeze(x2_organ) #out shape [slices, n_outputs]
        x4 = x3.transpose(0,1).unsqueeze(0) #out shape [1, n_outputs, slices]
        x5 = self.avgpool_1d(x4) #out shape [1, n_outputs, 1]
        x6 = torch.squeeze(x5, dim=2) #out shape [1, n_outputs]
        return x6, x4, attn #final_pred, slice_pred, attn
    
    def forward(self, x):
        #Extract features using the same ResNet and 3D conv layers:
        #Note that x is a dictionary with keys right_lung, heart, and left_lung
        rl_final_pred, rl_slice_pred, rl_attn  = self.extract_organ_representation(x['right_lung'], self.conv3_lung)
        h_final_pred, h_slice_pred, h_attn = self.extract_organ_representation(x['heart'], self.conv3_heart)
        ll_final_pred, ll_slice_pred, ll_attn = self.extract_organ_representation(x['left_lung'], self.conv3_lung)
        
        #Now concatenate to get the final label vector
        #Order: heart, left_lung, right_lung (lexicographic because that is the
        #order of the ground truth)
        return torch.cat((h_final_pred,ll_final_pred,rl_final_pred),1)
    