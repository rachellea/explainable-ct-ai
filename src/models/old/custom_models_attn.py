#custom_models_attn.py
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

class Body_Fully2D_83Avg_HFAttn(nn.Module):
    """Model for big data.
    Fully 2D convolutional on slices. Get 83 probabilities for each slice
    and average them together to get the final prediction
    Includes Height Feature Attention"""
    def __init__(self, n_outputs, use_attn_features, use_attn_height):
        super(Body_Fully2D_83Avg_HFAttn, self).__init__()
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
            
            nn.Conv2d(16, n_outputs, kernel_size = (2,2), stride=(2,2), padding=0)).to('cuda:0')
        
        self.avgpool = nn.AvgPool1d(kernel_size=self.num_slices).to('cuda:0')
        #TODO try with max pool or other kinds of pooling
        
    def forward(self, x):
        x = x.to('cuda:1')
        shape = list(x.size())
        batch_size = int(shape[0])
        assert batch_size == 1
        x = x.view(batch_size*self.num_slices,3,420,420)
        x = self.features(x) # out shape [134,512,14,14]
        x = x.to('cuda:0')
        
        if self.use_attn_features:
            af = self.sigmoid(self.attn_features)
            x = torch.mul(x, af)
        
        if self.use_attn_height:
            ah = self.sigmoid(self.attn_height)
            x = torch.mul(x, ah)
        
        x = self.conv2d_reducing(x) #out shape [134,83,1,1]
        x = torch.squeeze(x) #out shape [134, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, 134]
        x = self.avgpool(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class Body_Fully2D_83Avg_HFAttn_Spatial(nn.Module):
    """Model for big data.
    Fully 2D convolutional on slices. Get 83 probabilities for each slice
    and average them together to get the final prediction
    Includes Height Feature Attention
    Includes spatial attention"""
    def __init__(self, n_outputs, use_attn_features, use_attn_height):
        super(Body_Fully2D_83Avg_HFAttn_Spatial, self).__init__()
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
        
        #conv2d input [134,512,14,14]
        self.conv2d_reducing1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True)).to('cuda:0')
        
        self.attn_branch = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0)).to('cuda:0')
        
        self.conv2d_reducing2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (2,2), stride=(2,2), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (2,2), stride=(2,2), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, n_outputs, kernel_size = (2,2), stride=(2,2), padding=0)).to('cuda:0')
        
        self.avgpool = nn.AvgPool1d(kernel_size=self.num_slices).to('cuda:0')
        #TODO try with max pool or other kinds of pooling
        
    def forward(self, x):
        x = x.to('cuda:1')
        shape = list(x.size())
        batch_size = int(shape[0])
        assert batch_size == 1
        x = x.view(batch_size*self.num_slices,3,420,420)
        x = self.features(x) # out shape [134,512,14,14]
        x = x.to('cuda:0')
        
        #Height and/or Feature Attention
        if self.use_attn_features:
            af = self.sigmoid(self.attn_features)
            x = torch.mul(x, af)
        
        if self.use_attn_height:
            ah = self.sigmoid(self.attn_height)
            x = torch.mul(x, ah)
        
        #Convolutions and Spatial Attention
        x1 = self.conv2d_reducing1(x)
        attn = self.sigmoid(self.attn_branch(x))
        x1_times_attn = torch.mul(x1,attn)
        x2 = self.conv2d_reducing2(x1_times_attn) #out shape [134,83,1,1]
        
        #Average Pooling
        x3 = torch.squeeze(x2) #out shape [134, 83]
        x4 = x3.transpose(0,1).unsqueeze(0) #out shape [1, 83, 134]
        x5 = self.avgpool(x4) #out shape [1, 83, 1]
        x6 = torch.squeeze(x5, dim=2) #out shape [1, 83]
        return x6

class ResNet18_TripleCrop_Attn(nn.Module):
    #TODO: update this to use the BodyFully2D Model as baseline!
    """Model for big data. ResNet18 then 3D conv then FC.
    Here the input x is a Python dictionary with keys right_lung, heart, and
    left_lung.
    This model applies the same ResNet and 3D conv feature extractor to each crop,
    then applies separate FC layers to make the final predictions.
    Then this model uses attention. 
    Note that running this model requires model parallelism because it does not
    fit on a single GPU."""
    def __init__(self, n_outputs_lung, n_outputs_heart):
        super(ResNet18_TripleCrop_Attn, self).__init__()
        
        #Note that x = x.to('cuda:0') means x will be sent to the first GPU
        #that torch can see. So if you've done CUDA_VISIBLE_DEVICES=2,3 then
        #'cuda:0' may correspond to GPU 2 in reality.
        resnet = models.resnet18(pretrained=True)
        ##in total the resnet has 10 children. You are going to use 0:-2 of them
        self.features1 = nn.Sequential(*(list(resnet.children())[0:5])).to('cuda:1')
        self.features2 = nn.Sequential(*(list(resnet.children())[5:-2])).to('cuda:0')
        
        #Now use resnet again to get a branch that will calculate attention
        resnet_reloaded = models.resnet18(pretrained=True)
        self.sigmoid = nn.Sigmoid()
        self.attn_branch = nn.Sequential(*(list(resnet_reloaded.children())[5:-2])).to('cuda:0')
        
        #conv input torch.Size[1,134,512,10,5]
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(134, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True)).to('cuda:0')
        
        self.classifier_heart = nn.Sequential(
            nn.Linear(16*18*5*1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs_heart)).to('cuda:0')
        
        self.classifier_lung = nn.Sequential(
            nn.Linear(16*18*5*1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs_lung)).to('cuda:0')
            
    def extract_organ_representation(self, z):
        """z is a crop from the full volume representing the right_lung, heart,
        or left_lung"""
        #z input shape is [1, 134, 3, 420, 210]
        z = z.to('cuda:1')
        shape = list(z.size())
        batch_size = int(shape[0])
        z = z.view(batch_size*134,3,420,210)
        z = self.features1(z)
        z = z.to('cuda:0')
        z2 = self.features2(z) #after features: [134, 512, 14, 7]
        attn = self.sigmoid(self.attn_branch(z)) #after attn: [134, 512, 14, 7]
        z2_times_attn = torch.mul(z2,attn)
        z3 = z2_times_attn.unsqueeze(0) #after unsqueeze: [1, 134, 512, 14, 7]
        z3 = self.reducingconvs(z3) #after convs: [1, 16, 18, 5, 1]
        z3 = z3.view(1, 16*18*5*1)
        return z3
    
    def forward(self, x):
        #Extract features using the same ResNet and 3D conv layers:
        #Note that x is a dictionary with keys right_lung, heart, and left_lung
        right_lung = self.extract_organ_representation(x['right_lung'])
        heart = self.extract_organ_representation(x['heart'])
        left_lung = self.extract_organ_representation(x['left_lung'])
        
        #Perform classification that is specific to the organ
        #Lungs share the classifier. Heart has a separate classifier.
        right_lung = self.classifier_lung(right_lung)
        heart = self.classifier_heart(heart)
        left_lung = self.classifier_lung(left_lung)
        
        #Now concatenate to get the final label vector
        #Order: heart, left_lung, right_lung (lexicographic because that is the
        #order of the ground truth)
        return torch.cat((heart,left_lung,right_lung),1)

class ResNet18_Batch_Body3DConv_Attn(nn.Module):
    #TODO: update this to use the BodyFully2D Model as baseline!
    """Model for big data. ResNet18 then 3D conv then FC.
    3d conv used here follows recommendation to reshape to
    [batch_size, 512, 134, 14, 14] before performing 3d conv so that the
    convolution occurs over 'mini-bodies.'
    The 134 represents the height of the body, and the 14 x 14 represents the
    result of applying a particular ResNet filter (one of the 512 different
    filters.)
    Includes attention mechanism."""
    def __init__(self, n_outputs):
        super(ResNet18_Batch_Body3DConv_Attn, self).__init__()        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2])).to('cuda:1')
        
        #conv input torch.Size([1,512,134,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0), #different: changed in_channels from 134 to 512
            nn.ReLU(inplace=True)).to('cuda:0')
        
        self.sigmoid = nn.Sigmoid()
        self.attn_branch = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0), #different: changed in_channels from 134 to 512
            nn.ReLU(inplace=True)).to('cuda:0')
        
        self.reducingconvs2 = nn.Sequential(nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True)).to('cuda:0')
        
        self.classifier = nn.Sequential(
            nn.Linear(16*4*5*5, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs)).to('cuda:0')
        
    def forward(self, x):
        x = x.to('cuda:1')
        shape = list(x.size())
        batch_size = int(shape[0])
        x = x.view(batch_size*134,3,420,420) #DD: checked
        x = self.features(x)
        x = x.to('cuda:0')
        
        assert batch_size == 1 #this is correct only for batch_size==1
        x = x.transpose(0,1).unsqueeze(0)
        x = x.contiguous()
        x2 = self.reducingconvs(x)
        attn = self.sigmoid(self.attn_branch(x))
        x2_times_attn = torch.mul(x2,attn)
        x3 = self.reducingconvs2(x2_times_attn)
        
        #output is shape [batch_size, 16, 4, 5, 5]
        x3 = x3.view(batch_size, 16*4*5*5)
        x3 = self.classifier(x3)
        return x3

class Fully2D_HFAttn_Spatial_Expanded(nn.Module): #Body, 83Avg
    """Model for big data.
    Fully 2D convolutional on slices. Get 83 probabilities for each slice
    and average them together to get the final prediction
    Includes Height Feature Attention
    Includes spatial attention
    Expanded: the convolutions are padded and have smaller stride so that
    feature maps remain at 14 x 14"""
    def __init__(self, n_outputs, use_attn_features, use_attn_height):
        super(Fully2D_HFAttn_Spatial_Expanded, self).__init__()
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
        
        #conv2d input [134,512,14,14]
        self.conv2d_reducing1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True)).to('cuda:0')
        
        self.attn_branch = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=1)).to('cuda:0')
        
        self.conv2d_reducing2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, kernel_size = (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, n_outputs, kernel_size = (3,3), stride=(1,1), padding=1)).to('cuda:0')
        
        self.avgpool_2d = nn.AvgPool2d(kernel_size=(14,14)).to('cuda:0')
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.num_slices).to('cuda:0')
        
    def forward(self, x):
        x = x.to('cuda:1')
        shape = list(x.size())
        batch_size = int(shape[0])
        assert batch_size == 1
        x = x.view(batch_size*self.num_slices,3,420,420)
        x = self.features(x) # out shape [134,512,14,14]
        x = x.to('cuda:0')
        
        #Height and/or Feature Attention
        if self.use_attn_features:
            af = self.sigmoid(self.attn_features)
            x = torch.mul(x, af)
        
        if self.use_attn_height:
            ah = self.sigmoid(self.attn_height)
            x = torch.mul(x, ah)
        
        #Convolutions and Spatial Attention
        x1 = self.conv2d_reducing1(x)
        attn = self.sigmoid(self.attn_branch(x))
        x1_times_attn = torch.mul(x1,attn)
        x2 = self.conv2d_reducing2(x1_times_attn) #out shape [134,83,14,14]
        
        #Average Pooling
        x2_pooled = self.avgpool_2d(x2) #out shape [134,83,1,1]
        x3 = torch.squeeze(x2_pooled) #out shape [134, 83]
        x4 = x3.transpose(0,1).unsqueeze(0) #out shape [1, 83, 134]
        x5 = self.avgpool_1d(x4) #out shape [1, 83, 1]
        x6 = torch.squeeze(x5, dim=2) #out shape [1, 83]
        return x6