#custom_models.py
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

class ResNet18_Original(nn.Module):
    """Model for big data. ResNet18 then 3D conv then FC."""
    def __init__(self, n_outputs):
        super(ResNet18_Original, self).__init__()        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #conv input torch.Size([1,134,512,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(134, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True)) #almost the same as AlexNet except 3,2,2 here instead of 2,2,2
        
        self.classifier = nn.Sequential(
            nn.Linear(16*18*5*5, 128), #7200
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs))
        
    def forward(self, x):
        shape = list(x.size())
        #example shape: [1,134,3,420,420]
        #example shape: [2,134,3,420,420]
        #Can't use fixed batch size because it could theoretically change at the end of the set
        #(although probably not here since I can only do batch size of 1 per GPU)
        batch_size = int(shape[0])
        x = x.view(batch_size*134,3,420,420)
        x = self.features(x)
        x = x.view(batch_size,134,512,14,14)
        x = self.reducingconvs(x)
        #output is shape [batch_size, 16, 18, 5, 5]
        x = x.view(batch_size, 16*18*5*5)
        x = self.classifier(x)
        return x


class ResNet18_Body(nn.Module):
    """Model for big data. ResNet18 then 3D conv then FC.
    3d conv used here follows recommendation to reshape to
    [batch_size, 512, 134, 14, 14] before performing 3d conv so that the
    convolution occurs over 'mini-bodies.'
    The 134 represents the height of the body, and the 14 x 14 represents the
    result of applying a particular ResNet filter (one of the 512 different
    filters.)"""
    def __init__(self, n_outputs):
        super(ResNet18_Body, self).__init__()        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #conv input torch.Size([1,512,134,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0), #different: changed in_channels from 134 to 512
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True)) #almost the same as AlexNet except 3,2,2 here instead of 2,2,2
        
        self.classifier = nn.Sequential(
            nn.Linear(16*4*5*5, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs))
        
    def forward(self, x):
        shape = list(x.size())
        batch_size = int(shape[0])
        x = x.view(batch_size*134,3,420,420) #DD: checked
        x = self.features(x)
        
        #Note: doing the following is WRONG and does NOT result in mini-bodies
        #representation: x = x.view(batch_size,512,134,14,14)
        # DD: the right way to do it:
        assert batch_size == 1 #this is correct only for batch_size==1
        x = x.transpose(0,1).unsqueeze(0)
        x = x.contiguous()
        x = self.reducingconvs(x)
        
        #output is shape [batch_size, 16, 4, 5, 5]
        x = x.view(batch_size, 16*4*5*5)
        x = self.classifier(x)
        return x


class ResNet18_Body_1x1(nn.Module):
    """Model for big data. ResNet18 then 3D conv then FC.
    Uses a 2d convolution to collapse the 512 feature maps into one before the
    3d convolution step."""
    def __init__(self, n_outputs):
        super(ResNet18_Body_1x1, self).__init__()        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2])).to('cuda:1')
        
        #conv2d input torch.Size([134,512,14,14])
        self.conv2d_512to1 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size = (1,1), stride=(1,1), padding=0),
            nn.ReLU(inplace=True)).to('cuda:0') #out [134,1,14,14]
        
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
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
        x = x.view(batch_size*134,3,420,420)
        x = self.features(x)
        x = x.to('cuda:0')
        #shape [134,512,14,14]
        x = self.conv2d_512to1(x)
        #shape [134,1,14,14]
        assert batch_size == 1
        x = x.transpose(0,1).unsqueeze(0)
        x = x.contiguous()
        #shape [1,1,134,14,14]
        x = self.reducingconvs(x)
        x = x.view(batch_size, 16*4*5*5)
        x = self.classifier(x)
        return x


class ResNet18_Body_2DConv(nn.Module):
    """Model for big data.
    Uses a 2d convolution to collapse the 512 feature maps gradually. Then
    a little bit of 3d convolution."""
    def __init__(self, n_outputs):
        super(ResNet18_Body_2DConv, self).__init__()        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #conv2d input [134,512,14,14]
        self.conv2d_reducing = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size = (3,3), stride=(1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size = (2,2), stride=(2,2), padding=0),
            nn.ReLU(inplace=True))
        
        #conv3d input [1, 16, 134, 5, 5]
        self.conv3d_reducing = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size = (3,1,1), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(16, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True))
        
        self.classifier = nn.Sequential(
            nn.Linear(16*14*2*2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs))
        
    def forward(self, x):
        shape = list(x.size())
        batch_size = int(shape[0])
        x = x.view(batch_size*134,3,420,420)
        x = self.features(x) # out shape [134,512,14,14]
        x = self.conv2d_reducing(x) #out shape [134,16,5,5]
        assert batch_size == 1
        x = x.transpose(0,1).unsqueeze(0) #[1, 16, 134, 5, 5]
        x = x.contiguous()
        x = self.conv3d_reducing(x)
        x = x.view(batch_size, 16*14*2*2)
        x = self.classifier(x)
        return x


class ResNet18_Body_Fully2D(nn.Module):
    """Model for big data.
    Fully 2D convolutional on slices. Interactions between slices happen
    at the fully connected stage."""
    def __init__(self, n_outputs):
        super(ResNet18_Body_Fully2D, self).__init__()        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2])).to('cuda:1')
        
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
            
            nn.Conv2d(16, 16, kernel_size = (2,2), stride=(2,2), padding=0),
            nn.ReLU(inplace=True)
            ).to('cuda:0')
        
        self.classifier = nn.Sequential(
            nn.Linear(134*16, 128),
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
        x = x.view(batch_size*134,3,420,420)
        x = self.features(x) # out shape [134,512,14,14]
        x = x.to('cuda:0')
        x = self.conv2d_reducing(x) #out shape [134,16,1,1]
        x = x.view(batch_size, 134*16)
        x = self.classifier(x)
        return x


class ResNet18_Body_Fully2D_83Avg(nn.Module):
    """Model for big data.
    Fully 2D convolutional on slices. Get 83 probabilities for each slice
    and average them together to get the final prediction"""
    def __init__(self, n_outputs):
        super(ResNet18_Body_Fully2D_83Avg, self).__init__()        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2])).to('cuda:1')
        
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
        
        self.avgpool = nn.AvgPool1d(kernel_size=134).to('cuda:0')
        #TODO try with max pool or other kinds of pooling
        
    def forward(self, x):
        x = x.to('cuda:1')
        shape = list(x.size())
        batch_size = int(shape[0])
        assert batch_size == 1
        x = x.view(batch_size*134,3,420,420)
        x = self.features(x) # out shape [134,512,14,14]
        x = x.to('cuda:0')
        x = self.conv2d_reducing(x) #out shape [134,83,1,1]
        x = torch.squeeze(x) #out shape [134, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, 134]
        x = self.avgpool(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class ResNet18_Body_Proj_3Slices(nn.Module):
    """First does max pooling to get a stack of maximum intensity projections
    Otherwise similar to the BodyConv model"""
    def __init__(self, n_outputs):
        super(ResNet18_Body_Proj_3Slices, self).__init__()
        self.pooling = nn.MaxPool3d(kernel_size = (3,1,1), stride=(3,1,1), padding=0)
        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0), #different: changed in_channels from 134 to 512
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True))
        
        self.classifier = nn.Sequential(
            nn.Linear(16*1*5*5, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs))
        
    def forward(self, x):
        #note: this model expects one-channel input to start: [1, 402, 420, 420]
        shape = list(x.size())
        batch_size = int(shape[0])
        assert batch_size == 1
        x = x.unsqueeze(0) 
        x = self.pooling(x) #out [1, 1, 134, 420, 420]
        x = torch.squeeze(x) #out [134, 420, 420]
        pad = (0,0,0,0,0,1)
        x = F.pad(x, pad, mode='constant', value=torch.min(x)) #out [135,420,420]
        x = x.view(45,3,420,420) #TODO CHECK THIS
        x = self.features(x)
        x = x.transpose(0,1).unsqueeze(0)
        x = x.contiguous() #out [1, 512, 45, 14, 14]
        x = self.reducingconvs(x) #out [1, 16, 1, 5, 5]
        x = x.view(batch_size, 16*1*5*5)
        x = self.classifier(x)
        return x


class ResNet18_Original_Proj_Slices(nn.Module):
    """Projections model based on Original model.
    <n_slices> is the number of slices in the input, e.g. 402
    <n_pool> is the number of slices that should get max pooled together at
        the beginning of the model."""
    def __init__(self, n_outputs, n_slices, n_pool):
        super(ResNet18_Original_Proj_Slices, self).__init__()
        self.pooling = nn.MaxPool3d(kernel_size = (n_pool,1,1), stride=(n_pool,1,1), padding=0)
        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #calculating channels:
        #for n_pool=1 this reduces to the Original model with n_channels=134
        #for n_pool=3, n_channels=45. For n_pool=5, n_channels=27.
        #For n_pool=7, n_channels=20.
        self.after_max_pool = int(n_slices/n_pool)
        if self.after_max_pool % 3 == 0:
            self.pad_amount = 0
            self.n_channels = int(self.after_max_pool/3)
        if (self.after_max_pool+1) % 3 == 0:
            self.pad_amount = 1
            self.n_channels = int((self.after_max_pool+1)/3)
        elif (self.after_max_pool+2) % 3 == 0:
            self.pad_amount = 2
            self.n_channels = int((self.after_max_pool+2)/3)
        
        #conv input torch.Size([1,45,512,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(self.n_channels, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True)) #almost the same as AlexNet except 3,2,2 here instead of 2,2,2
        
        self.classifier = nn.Sequential(
            nn.Linear(16*18*5*5, 128), #7200
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs))
        
    def forward(self, x):
        #note: this model expects one-channel input to start: [1, 402, 420, 420]
        shape = list(x.size())
        batch_size = int(shape[0])
        assert batch_size == 1
        x = x.unsqueeze(0) 
        x = self.pooling(x) #e.g. for n_pool=3, out [1, 1, 134, 420, 420]
        x = torch.squeeze(x) #e.g. out [134, 420, 420]
        pad = (0,0,0,0,0,self.pad_amount)
        x = F.pad(x, pad, mode='constant', value=torch.min(x)) #e.g. out [135,420,420]
        x = x.view(self.n_channels,3,420,420) #TODO CHECK THIS
        x = self.features(x)
        x = x.unsqueeze(0) #e.g. out [1, 45, 512, 14, 14]
        x = self.reducingconvs(x) #e.g. out [1, 16, 1, 5, 5]
        x = x.view(batch_size, 16*18*5*5)
        x = self.classifier(x)
        return x

class ResNet18_Original_Proj3_AxCorSag(nn.Module):
    """Projections model based on Original model
    Uses axial, sagittal, and coronal projections together.
    As a consequence, there is no data augmentation in this model."""
    def __init__(self, n_outputs):
        super(ResNet18_Original_Proj3_AxCorSag, self).__init__()
        self.pooling = nn.MaxPool3d(kernel_size = (3,1,1), stride=(3,1,1), padding=0)
        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #conv input torch.Size([1,47,512,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(47, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(16, 1, kernel_size = (1,1,1), stride=(1,1,1), padding=0),
            nn.ReLU(inplace=True))
        
        self.classifier = nn.Sequential(
            nn.Linear(450*3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs))
    
    def pooled_projection(self, tensor):
        """Return a pooled projection of the provided tensor"""
        #don't overwrite <tensor> because for axial, <tensor> is x
        temp = self.pooling(tensor) #out [1, 1, 140, 420, 420]
        temp = torch.squeeze(temp) #out [140, 420, 420]
        pad = (0,0,0,0,0,1)
        temp = F.pad(temp, pad, mode='constant', value=torch.min(tensor)) #for axial, out [141,420,420]
        return temp.view(47,3,420,420) #TODO CHECK THIS
    
    def get_all_pooled_projections(self, x):
        """Get axial, coronal, and sagittal projections from x"""
        x = x.unsqueeze(0) #out [1, 1, 420, 420, 420] = [1, 1, ax, cor, sag]
        
        #Pooled projections: axial
        axial = self.pooled_projection(x)
        
        #Pooled projections: coronal
        coronal = torch.transpose(torch.transpose(x,2,3), 3,4) #inner out [1, 1, cor, ax, sag], then [1, 1, cor, sag, ax]
        coronal = self.pooled_projection(coronal) #out [1, 1, 140, 420, 420]->[141,420,420]
        
        #Pooled projections: sagittal
        sagittal = torch.transpose(torch.transpose(x,2,4), 3,4) #inner out [1, 1, sag, cor, ax], then [1, 1, sag, ax, cor]
        sagittal = self.pooled_projection(sagittal)
        
        return axial, coronal, sagittal
    
    def get_pooled_projection_features(self, tensor):
        """Get features from the ResNet (self.features) and the reducing
        convs (self.reducingconvs) for the provided <tensor> which represents
        a pooled projection"""
        tensor = self.features(tensor)
        tensor = tensor.unsqueeze(0) #out [1, 47, 512, 14, 14]
        tensor = self.reducingconvs(tensor) #out [1, 1, 18, 5, 5]
        return tensor.view(1, 18*5*5)
        #Future TODO: try some clever convolutional way of combining the
        #axial, coronal, and sagittal representations for each of the disease
        #predictions, in an attention-like way
        
    def forward(self, x):
        #note: this model expects one-channel input to start: [1, 402, 420, 420]
        #where the dimensions are [axial slice, coronal slice, sagittal slice]
        shape = list(x.size())
        batch_size = int(shape[0])
        assert batch_size == 1
        
        axial, coronal, sagittal = self.get_all_pooled_projections(x)
        axial = self.get_pooled_projection_features(axial) #out [1, 450]
        coronal = self.get_pooled_projection_features(coronal) #out [1, 450]
        sagittal = self.get_pooled_projection_features(sagittal) #out [1, 450]
        return self.classifier(torch.cat([axial,coronal,sagittal],dim=1))

class ThreeDConv(nn.Module):
    """3D conv then FC. No pretrained model"""
    def __init__(self, n_outputs):
        super(ThreeDConv, self).__init__()        
        #conv input [1,1,402,420,420]
        # torch.nn.Conv2d or Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size = (3,3,3), stride=(3,3,3), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 128, kernel_size = (3,3,3), stride=(3,3,3), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(128, 256, kernel_size = (3,3,3), stride=(3,3,3), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(256, 512, kernel_size = (3,3,3), stride=(3,3,3), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(512, 128, kernel_size = (1,1,1), stride=(1,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(128, 64, kernel_size = (1,1,1), stride=(1,1,1), padding=0),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64*4*5*5, 128), #1024
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
                        
            nn.Linear(96, n_outputs))
        
    def forward(self, x):
        #input x has shape [1,402,420,420]
        shape = list(x.size())
        batch_size = int(shape[0])
        assert batch_size == 1
        x = x.unsqueeze(dim=0) #get shape [1,1,402,420,420] = [N,C,D,H,W]
        x = self.reducingconvs(x)
        x = x.view(1, 64*4*5*5)
        x = self.classifier(x)
        return x


class ResNet18_Original_Ablate_RandomInitResNet(nn.Module):
    """Model for big data. ResNet18 then 3D conv then FC.
    Initialize the ResNet randomly instead of pretrained."""
    def __init__(self, n_outputs):
        super(ResNet18_Original_Ablate_RandomInitResNet, self).__init__()
        print('ResNet18_Batch_Ablate_RandomInitResNet: pretrained=False')
        resnet = models.resnet18(pretrained=False)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(134, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True))
        
        self.classifier = nn.Sequential(
            nn.Linear(16*18*5*5, 128), #7200
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs))
        
    def forward(self, x):
        shape = list(x.size())
        batch_size = int(shape[0])
        x = x.view(batch_size*134,3,420,420)
        x = self.features(x)
        x = x.view(batch_size,134,512,14,14)
        x = self.reducingconvs(x)
        x = x.view(batch_size, 16*18*5*5)
        x = self.classifier(x)
        return x


class ResNet18_Original_Ablate_PoolInsteadOf3D(nn.Module):
    """Model for big data. ResNet18 then 3D conv then FC.
    Perform pooling instead of 3D convolution."""
    def __init__(self, n_outputs):
        super(ResNet18_Original_Ablate_PoolInsteadOf3D, self).__init__()        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #conv input torch.Size([1,134,512,14,14])
        #torch.nn.MaxPool3d(kernel_size, stride=None)
        self.reducingpools = nn.Sequential(
            nn.MaxPool3d(kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.MaxPool3d(kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.MaxPool3d(kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True))
        
        self.reducingpools2 = nn.Sequential(
            nn.MaxPool3d(kernel_size = (8,1,1), stride=(8,1,1), padding=0),
            nn.ReLU(inplace=True)
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(16*18*5*5, 128), #7200
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs))
        
    def forward(self, x):
        shape = list(x.size())
        batch_size = int(shape[0])
        x = x.view(batch_size*134,3,420,420)
        x = self.features(x)
        x = x.view(batch_size,134,512,14,14)
        x = self.reducingpools(x)
        #Output of reducingpools is [1, 134, 18, 5, 5]
        #We need it to be [1, 16, 18, 5, 5]
        #134 / 16 is 8.3. Do more pooling.
        assert batch_size == 1
        x = torch.squeeze(x) #size [134, 18, 5, 5]
        x = x.transpose(0,1) #size [18, 134, 5, 5]
        x = self.reducingpools2(x)
        #Output is [18, 16, 5, 5]
        x = x.transpose(0,1) #size [16, 18, 5, 5]
        x = x.unsqueeze(0) #size [1, 16, 18, 5, 5]
        x = x.contiguous()
        x = x.view(1, 16*18*5*5)
        x = self.classifier(x)
        return x


class ResNet18_Original_Seg(nn.Module):
    """Model for big data. ResNet18 then 3D conv then FC.
    Difference from ResNet18_Batch: this model uses smaller-sized inputs"""
    def __init__(self, n_outputs):
        super(ResNet18_Original_Seg, self).__init__()        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(107, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True))
        
        self.classifier = nn.Sequential(
            nn.Linear(16*18*3*3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs))
        
    def forward(self, x):
        shape = list(x.size())
        #example shape: [1,107,3,340,340]
        #example shape: [2,107,3,340,340]
        batch_size = int(shape[0])
        x = x.view(batch_size*107,3,340,340)
        x = self.features(x)
        x = x.view(batch_size,107,512,11,11)
        x = self.reducingconvs(x)
        x = x.view(batch_size, 16*18*3*3)
        x = self.classifier(x)
        return x