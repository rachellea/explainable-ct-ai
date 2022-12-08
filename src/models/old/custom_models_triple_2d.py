#custom_models_triple_2d.py
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

class Fully2D_HFAttn_Spatial_Triple(nn.Module): #Body, 83Avg
    """Model for big data.
    Fully 2D convolutional on slices. Get 83 probabilities for each slice
    and average them together to get the final prediction
    Includes Height Feature Attention
    Includes spatial attention
    Uses triple crop input"""
    def __init__(self, use_attn_features, use_attn_height,
                 n_outputs_lung, n_outputs_heart):
        super(Fully2D_HFAttn_Spatial_Triple, self).__init__()
        self.use_attn_features = use_attn_features
        self.use_attn_height = use_attn_height
        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2])).to('cuda:1')
        
        #determine num slices
        self.num_slices = 15 #45/3 (9 projections input data)
        
        #attention
        self.sigmoid = nn.Sigmoid().to('cuda:0')
        if self.use_attn_features:
            means_features = torch.Tensor(np.full((1,512,1,1),1.0,dtype='float'))
            stds_features = torch.Tensor(np.full((1,512,1,1),0.05,dtype='float'))
            self.attn_features = nn.Parameter(torch.normal(mean=means_features,std=stds_features),requires_grad=True).to('cuda:0')
        
        if self.use_attn_height:
            means_height = torch.Tensor(np.full((self.num_slices,1,1,1),1.0,dtype='float'))
            stds_height = torch.Tensor(np.full((self.num_slices,1,1,1),0.05,dtype='float'))
            self.attn_height = nn.Parameter(torch.normal(mean=means_height,std=stds_height),requires_grad=True).to('cuda:0')
        
        #Lung Convolutions
        #conv2d input [134,512,14,14]
        self.conv2d_reducing_lung1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True)).to('cuda:0')
        
        self.attn_branch_lung = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0)).to('cuda:0')
        
        self.conv2d_reducing_lung2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = (3,2), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (2,2), stride=(2,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (2,2), stride=(2,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, n_outputs_lung, kernel_size = (2,2), stride=(2,2), padding=0)).to('cuda:0')
        
        #Heart Convolutions
        self.conv2d_reducing_heart1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True)).to('cuda:0')
        
        self.attn_branch_heart = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0)).to('cuda:0')
        
        self.conv2d_reducing_heart2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = (3,2), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (2,2), stride=(2,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (2,2), stride=(2,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, n_outputs_heart, kernel_size = (2,2), stride=(2,2), padding=0)).to('cuda:0')
        
        self.avgpool = nn.AvgPool1d(kernel_size=self.num_slices).to('cuda:0')
    
    def extract_organ_representation(self, z, conv1, attnbranch, conv2):
        """z is a crop from the full volume representing the right_lung, heart,
        or left_lung"""
        z = z.to('cuda:1')
        shape = list(z.size())
        batch_size = int(shape[0])
        assert batch_size == 1
        z = z.view(batch_size*self.num_slices,3,420,210)
        z = self.features(z) # out shape [134,512,14,7]
        z = z.to('cuda:0')
        
        #Height and/or Feature Attention
        if self.use_attn_features:
            af = self.sigmoid(self.attn_features)
            z = torch.mul(z, af)
        
        if self.use_attn_height:
            ah = self.sigmoid(self.attn_height)
            z = torch.mul(z, ah)
        
        #Convolutions and Spatial Attention
        z1 = conv1(z)
        attn = self.sigmoid(attnbranch(z))
        z1_times_attn = torch.mul(z1,attn)
        z2 = conv2(z1_times_attn) #out shape [134,83,1,1]
        
        #Average Pooling
        z3 = torch.squeeze(z2) #out shape [134, 83]
        z4 = z3.transpose(0,1).unsqueeze(0) #out shape [1, 83, 134]
        z5 = self.avgpool(z4) #out shape [1, 83, 1]
        z6 = torch.squeeze(z5, dim=2) #out shape [1, 83]
        
        return z6, z4, attn #final_pred, slice_pred, attn
    
    def forward(self, x):
        #Extract features using the same ResNet and 3D conv layers:
        #Note that x is a dictionary with keys right_lung, heart, and left_lung
        rl_final_pred, rl_slice_pred, rl_attn  = self.extract_organ_representation(x['right_lung'],
                        self.conv2d_reducing_lung1,
                        self.attn_branch_lung, self.conv2d_reducing_lung2)
        h_final_pred, h_slice_pred, h_attn = self.extract_organ_representation(x['heart'],
                        self.conv2d_reducing_heart1,
                        self.attn_branch_heart, self.conv2d_reducing_heart2)
        ll_final_pred, ll_slice_pred, ll_attn = self.extract_organ_representation(x['left_lung'],
                        self.conv2d_reducing_lung1,
                        self.attn_branch_lung, self.conv2d_reducing_lung2)
        
        #Now concatenate to get the final label vector
        #Order: heart, left_lung, right_lung (lexicographic because that is the
        #order of the ground truth)
        return torch.cat((h_final_pred,ll_final_pred,rl_final_pred),1)

################################################################################
################################################################################
################################################################################
class Fully2D_HFAttn_Spatial_Triple2(nn.Module): #Body, 83Avg
    """Model for big data.
    Fully 2D convolutional on slices. Get 83 probabilities for each slice
    and average them together to get the final prediction
    Includes Height Feature Attention
    Includes spatial attention
    Uses triple crop input
    More parts of the network are shared between heart and lungs"""
    def __init__(self, use_attn_features, use_attn_height,
                 n_outputs_lung, n_outputs_heart):
        super(Fully2D_HFAttn_Spatial_Triple2, self).__init__()
        self.use_attn_features = use_attn_features
        self.use_attn_height = use_attn_height
        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2])).to('cuda:1')
        
        #determine num slices
        self.num_slices = 15 #45/3 (9 projections input data)
        
        #attention
        self.sigmoid = nn.Sigmoid().to('cuda:0')
        if self.use_attn_features:
            means_features = torch.Tensor(np.full((1,512,1,1),1.0,dtype='float'))
            stds_features = torch.Tensor(np.full((1,512,1,1),0.05,dtype='float'))
            self.attn_features = nn.Parameter(torch.normal(mean=means_features,std=stds_features),requires_grad=True).to('cuda:0')
        
        if self.use_attn_height:
            means_height = torch.Tensor(np.full((self.num_slices,1,1,1),1.0,dtype='float'))
            stds_height = torch.Tensor(np.full((self.num_slices,1,1,1),0.05,dtype='float'))
            self.attn_height = nn.Parameter(torch.normal(mean=means_height,std=stds_height),requires_grad=True).to('cuda:0')
        
        #Lung Convolutions
        #conv2d input [134,512,14,14]
        self.conv2d_reducing1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True)).to('cuda:0')
        
        self.attn_branch = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0)).to('cuda:0')
        
        self.conv2d_reducing2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = (3,2), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (2,2), stride=(2,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (2,2), stride=(2,1), padding=0),
            nn.ReLU(inplace=True)).to('cuda:0')
        
        #Lung
        self.conv2d_final_lung = nn.Sequential(
            nn.Conv2d(16, n_outputs_lung, kernel_size = (2,2), stride=(2,2), padding=0)).to('cuda:0')
        
        #Heart
        self.conv2d_final_heart = nn.Sequential(
            nn.Conv2d(16, n_outputs_heart, kernel_size = (2,2), stride=(2,2), padding=0)).to('cuda:0')
        
        self.avgpool = nn.AvgPool1d(kernel_size=self.num_slices).to('cuda:0')
    
    def extract_organ_representation(self, z, convfinal):
        """z is a crop from the full volume representing the right_lung, heart,
        or left_lung"""
        z = z.to('cuda:1')
        shape = list(z.size())
        batch_size = int(shape[0])
        assert batch_size == 1
        z = z.view(batch_size*self.num_slices,3,420,210)
        z = self.features(z) # out shape [134,512,14,7]
        z = z.to('cuda:0')
        
        #Height and/or Feature Attention
        if self.use_attn_features:
            af = self.sigmoid(self.attn_features)
            z = torch.mul(z, af)
        
        if self.use_attn_height:
            ah = self.sigmoid(self.attn_height)
            z = torch.mul(z, ah)
        
        #Convolutions and Spatial Attention
        z1 = self.conv2d_reducing1(z)
        attn = self.sigmoid(self.attn_branch(z))
        z1_times_attn = torch.mul(z1,attn)
        z2 = self.conv2d_reducing2(z1_times_attn) #out shape [134,83,1,1]
        
        #Organ-specific final convolution layer
        z2_organ = convfinal(z2)
        
        #Average Pooling
        z3 = torch.squeeze(z2_organ) #out shape [134, 83]
        z4 = z3.transpose(0,1).unsqueeze(0) #out shape [1, 83, 134]
        z5 = self.avgpool(z4) #out shape [1, 83, 1]
        z6 = torch.squeeze(z5, dim=2) #out shape [1, 83]
        
        return z6, z4, attn #final_pred, slice_pred, attn
    
    def forward(self, x):
        #Extract features using the same ResNet and 3D conv layers:
        #Note that x is a dictionary with keys right_lung, heart, and left_lung
        rl_final_pred, rl_slice_pred, rl_attn  = self.extract_organ_representation(x['right_lung'],
                        self.conv2d_final_lung)
        h_final_pred, h_slice_pred, h_attn = self.extract_organ_representation(x['heart'],
                        self.conv2d_final_heart)
        ll_final_pred, ll_slice_pred, ll_attn = self.extract_organ_representation(x['left_lung'],
                        self.conv2d_final_lung)
        
        #Now concatenate to get the final label vector
        #Order: heart, left_lung, right_lung (lexicographic because that is the
        #order of the ground truth)
        return torch.cat((h_final_pred,ll_final_pred,rl_final_pred),1)
    