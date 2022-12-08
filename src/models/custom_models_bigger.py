#custom_models_bigger.py
#Copyright (c) 2021 Rachel Lea Ballantyne Draelos

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

import torch, torch.nn as nn
from torchvision import models

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

#Import components
from . import components as cts

#AxialNet model but with more features in the custom convs
class AxialNetBigger_Variant2(nn.Module):
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, slices):
        super(AxialNetBigger_Variant2, self).__init__()
        self.n_outputs = n_outputs
        self.slices = slices #equal to 15 for 9 projections
        self.features = cts.resnet_features()
        self.conv2d = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size = (3,3), stride=(1,1), padding=0))
        
        self.fc = nn.Conv2d(512, n_outputs, kernel_size = (10,10), stride=(10,10), padding=0) #NOTE (10,10) instead of (6,6)!!!
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
    
    def forward(self, x):
        assert list(x.shape)==[1,self.slices,3,420,420]
        x = x.squeeze() #out shape [slices,3,420,420]
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x)
        x = self.fc(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

#More custom convolutional layers, with smaller kernel size to preserve the
#size of the feature maps
class AxialNetBigger_Variant3(nn.Module):
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, slices):
        super(AxialNetBigger_Variant3, self).__init__()
        self.n_outputs = n_outputs
        self.slices = slices #equal to 15 for 9 projections
        self.features = cts.resnet_features()
        self.conv2d = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = (1,1), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size = (1,1), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size = (1,1), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size = (1,1), stride=(1,1), padding=0))
        
        self.fc = nn.Conv2d(512, n_outputs, kernel_size = (14,14), stride=(14,14), padding=0) #NOTE (14,14) instead of (6,6)!!!
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
    
    def forward(self, x):
        assert list(x.shape)==[1,self.slices,3,420,420]
        x = x.squeeze() #out shape [slices,3,420,420]
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x)
        x = self.fc(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class AxialNetBigger_Variant2_Mask(nn.Module):
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 512, 6, 6] (way more features than original AxialNet)
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, slices):
        super(AxialNetBigger_Variant2_Mask, self).__init__()
        self.n_outputs = n_outputs
        self.slices = slices #equal to 15 for 9 projections
        self.features = cts.resnet_features()
        self.conv2d = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size = (3,3), stride=(1,1), padding=0))
        
        self.fc = nn.Conv2d(512, n_outputs, kernel_size = (10,10), stride=(10,10), padding=0) #NOTE (10,10) instead of (6,6)!!!
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
    
    def forward(self, x):
        assert list(x.shape)==[1,self.slices,3,420,420]
        x1 = x.squeeze() #out shape [slices,3,420,420]
        x1 = self.features(x1) #out shape [slices,512,14,14]
        x1f = self.conv2d(x1) #out shape [slices, 512, 10, 10] NOTE THIS IS BIGGER THAN 6,6 OF ORIGINAL AXIALNET
        x2 = self.fc(x1f) #out shape [slices,n_outputs,1,1]
        x2 = torch.squeeze(x2) #out shape [slices, n_outputs]
        x2_perslice_scores = x2.transpose(0,1).unsqueeze(0) #out shape [1, n_outputs, slices]
        x2 = self.avgpool_1d(x2_perslice_scores) #out shape [1, n_outputs, 1]
        x2f = torch.squeeze(x2, dim=2) #out shape [1, n_outputs]
        
        #Now calculate what the disease specific representation is in the
        #intermediate calculation of the fc layer.
        #First, make n_outputs copies of the slices x 512 x 6 x 6 representation:
        x1_repeated = x1f.repeat(self.n_outputs,1,1,1,1) #out shape [n_outputs, slices, 512, 6, 6]
        #Now select the fc_weights:
        fc_weights = self.fc.weight #shape [132, 512, 6, 6], where 132 is n_outputs
        fc_weights_unsq = fc_weights.unsqueeze(dim=1) #out shape [n_outputs, 1, 512, 6, 6]
        #Now multiply element wise. Broadcasting will occur.
        #we have [n_outputs, slices, 512, 6, 6] x [n_outputs, 1, 512, 6, 6]
        disease_reps = torch.mul(x1_repeated, fc_weights_unsq) #out shape [n_outputs, slices, 512, 6, 6]
        
        out = {'out':x2f,
               'x_perslice_scores':x2_perslice_scores,
               'disease_reps':disease_reps}
        return out

#Super simple model that just uses a ResNet with the global average pooling
#step followed by 1 FC layer. (In this setup, CAM, HiResCAM, and GradCAM are
#totally equivalent.)
class JustUseAvgPoolAfterResNet(nn.Module):
    """(1) ResNet18 [slices, 512, 1, 1]
       (2) FC layer to [n_outputs, 1, 1]"""
    def __init__(self, n_outputs, slices):
        super(JustUseAvgPoolAfterResNet, self).__init__()
        self.slices = slices #equal to 15 for 9 projections\
        resnet = models.resnet18(pretrained=True)
        self.features_and_pool = nn.Sequential(*(list(resnet.children())[:-1])) #up to -1 instead of up to -2
        self.fc = nn.Linear(512*slices, n_outputs)
        
    def forward(self, x):
        assert list(x.shape)==[1,self.slices,3,420,420]
        x = x.squeeze() #out shape [slices,3,420,420]
        x = self.features_and_pool(x) #out shape [slices,512,1,1]
        x = x.flatten().unsqueeze(0) #out shape [1,slices*512]
        x = self.fc(x)
        return x

class BodyConvBigger_Variant1(nn.Module): 
    """BodyConv-like model with more features in the reducingconvs followed by
    gloal average pooling and a single FC layer."""
    def __init__(self, n_outputs, slices):
        super(BodyConvBigger_Variant1, self).__init__()
        self.slices = slices
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #conv input torch.Size([1,512,134,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size = (3,3,3), stride=(3,3,3), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(512, 512, kernel_size = (3,3,3), stride=(3,3,3), padding=0))
        
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        
        self.classifier = nn.Sequential(
            nn.Linear(512, n_outputs))
        
    def forward(self, x):
        #batch size is 1
        assert list(x.shape)==[1,self.slices,3,420,420]
        x = x.squeeze() #out shape [slices,3,420,420]
        x = self.features(x)
        x = x.transpose(0,1).unsqueeze(0)
        x = x.contiguous()
        x = self.reducingconvs(x)
        x = self.pool(x)
        x = x.flatten().unsqueeze(0)
        x = self.classifier(x)
        return x


#Other models to try:
#CTNet model that uses more features in the custom convs
#CTNet model that uses more features in the custom convs followed by GAP and
#then a single FC layer. 