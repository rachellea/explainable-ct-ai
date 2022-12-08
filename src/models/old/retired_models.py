#retired_models.py
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

class GoogleNet(nn.Module):
    """GoogleNet then 3D conv then FC"""
    def __init__(self, n_outputs):
        super(GoogleNet, self).__init__()
        googlenet = models.googlenet(pretrained=True)
        self.features = nn.Sequential(*(list(googlenet.children())[:-6]))

        #conv input torch.Size([1, 140, 832, 18, 18])
        #this is as close to the AlexNet reducingconvs as possible except I
        #have to start out with some pooling due to memory issues
        self.reducingconvs = nn.Sequential(
            nn.MaxPool3d(kernel_size = (4,2,2), stride = (4,2,2)), #~[140,208,9,9]
            
            nn.Conv3d(140, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0), #out [6, 64, 85, 6, 6]
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0), #out [6, 32, 28, 4, 4]
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (2,2,2), stride=(2,2,2), padding=0), #out [6, 16, 14, 2, 2]
            nn.ReLU(inplace=True))
        
        self.classifier = nn.Sequential(
            nn.Linear(16*11*2*2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        
            nn.Linear(64, n_outputs))
        
    def forward(self, x):
        #input x has shape [6*140,3,H,W]
        x = self.features(x)
        shape = list(x.size())
        batch_size = int(shape[0]/140)
        x = x.view(batch_size, 140, 832, 18, 18)
        x = self.reducingconvs(x)
        #output is shape torch.Size([1, 16, 11, 2, 2])
        x = x.view(batch_size, 16*11*2*2)
        x = self.classifier(x)
        return x

class ResNet18_Batch_OneFilt(nn.Module):
    """Model for big data. ResNet18 then 3D conv then avgpool then a single FC.
    'OneFilt' in the name refers to the fact that this uses 1 x 1 x 1 convolution
    at the beginning of the 3D convolution stage. This architecture is
    conceptually similar to the CAM architecture. I want to see if it works."""
    def __init__(self, n_outputs):
        super(ResNet18_Batch_OneFilt, self).__init__()
        self.n_outputs = n_outputs
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #conv input torch.Size([1,512,134,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size = (1,1,1), stride=(1,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(256, 128, kernel_size = (1,1,1), stride=(1,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(128, 128, kernel_size = (1,1,1), stride=(1,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(128, n_outputs, kernel_size = (1,1,1), stride=(1,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(n_outputs, n_outputs, kernel_size = (1,1,1), stride=(1,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(n_outputs, n_outputs, kernel_size = (1,1,1), stride=(1,1,1), padding=0),
            
            #[1,n_outputs,134,14,14]
            nn.MaxPool3d(kernel_size = (134,14,14), stride=(134,14,14), padding=0))
        
    def forward(self, x):
        shape = list(x.size())
        #example shape: [1,134,3,420,420]
        #example shape: [2,134,3,420,420]
        #Can't use fixed batch size because it could theoretically change at the end of the set
        #(although probably not here since I can only do batch size of 1 per GPU)
        batch_size = int(shape[0])
        x = x.view(batch_size*134,3,420,420)
        x = self.features(x)
        x = x.view(batch_size,512,134,14,14) #This line is different from ResNet18_Batch() model
        x = self.reducingconvs(x)
        x = x.view(batch_size, self.n_outputs)
        return x

class ResNet18_Bottleneck(nn.Module):
    """Model for big data. ResNet18 then 3D conv then FC."""
    def __init__(self, n_outputs, bottleneck, bottleneck_size):
        super(ResNet18, self).__init__()        
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
            nn.Linear(16*18*3*3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))
        
        if bottleneck:
            self.classifier2 = nn.Sequential(
                 nn.Linear(64, bottleneck_size), #Bottleneck down to bottleneck_size, e.g. 16
                 nn.ReLU(inplace=True),
                 nn.Dropout(0.5),
                 
                 nn.Linear(bottleneck_size, n_outputs))
        else:
            self.classifier2 = nn.Sequential(
                nn.Linear(64, n_outputs))
        
    def forward(self, x):
        #put it in a format that can be used in the ResNet:
        x = x.view(134,3,308,308)
        x = self.features(x)
        x = x.view(1,134,512,10,10)
        x = self.reducingconvs(x)
        #output is shape [1, 16, 18, 3, 3]
        x = x.view(1, 16*18*3*3)
        x = self.classifier(x)
        x = self.classifier2(x)
        return x


class ResNet18_TripleCrop(nn.Module):
    """Model for big data. ResNet18 then 3D conv then FC.
    Here the input x is a Python dictionary with keys right_lung, heart, and
    left_lung.
    This model applies the same ResNet and 3D conv feature extractor to each crop,
    then applies separate FC layers to make the final predictions."""
    def __init__(self, n_outputs_lung, n_outputs_heart):
        super(ResNet18_TripleCrop, self).__init__()        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #conv input torch.Size[1,134,512,10,5]
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(134, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,1), stride=(3,2,1), padding=0),
            nn.ReLU(inplace=True)) #almost the same as ResNet18 except 3,2,1 here instead of 3,2,2
        
        self.classifier_heart = nn.Sequential(
            nn.Linear(16*18*3*1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(64, n_outputs_heart))
        
        self.classifier_lung = nn.Sequential(
            nn.Linear(16*18*3*1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(64, n_outputs_lung))
            
    def extract_organ_representation(self, z):
        """z is a crop from the full volume representing the right_lung, heart,
        or left_lung"""
        #z input shape is [1, 134, 3, 308, 154]
        z = self.features(z)
        z = z.view(1,134,512,10,5)
        z = self.reducingconvs(z)
        #output is shape [1, 16, 18, 3, 1]
        z = z.view(1, 16*18*3*1)
        return z
    
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

class ResNet18_LocDis_Flat(nn.Module):
    """Model for predicting flat location x disease output vector
    Only difference is that this doesn't explicitly go down to dimension 64,
    since the location x disease output vector is bigger than that (it's
    size 96)"""
    def __init__(self, n_outputs, bottleneck, bottleneck_size):
        super(ResNet18_LocDis_Flat, self).__init__()        
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
            nn.Linear(16*18*3*3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, n_outputs), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))
        
        if bottleneck:
            self.classifier2 = nn.Sequential(
                 nn.Linear(n_outputs, bottleneck_size), #Bottleneck down to bottleneck_size, e.g. 16
                 nn.ReLU(inplace=True),
                 nn.Dropout(0.5),
                 
                 nn.Linear(bottleneck_size, n_outputs))
        else:
            self.classifier2 = nn.Sequential(
                nn.Linear(n_outputs, n_outputs))
        
    def forward(self, x):
        #put it in a format that can be used in the ResNet:
        x = x.view(134,3,308,308)
        x = self.features(x)
        x = x.view(1,134,512,10,10)
        x = self.reducingconvs(x)
        #output is shape [1, 16, 18, 3, 3]
        x = x.view(1, 16*18*3*3)
        x = self.classifier(x)
        x = self.classifier2(x)
        return x


class ResNet34(nn.Module):
    """ResNet34 then 3D conv then FC"""
    def __init__(self, n_outputs):
        super(ResNet34, self).__init__()
        #note that resnet50 is too big and produces a memory error
        resnet = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #conv input torch.Size([1, 140, 512, 10, 10])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(140, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True)) #almost the same as AlexNet except 3,2,2 here instead of 2,2,2
        
        self.classifier = nn.Sequential(
            nn.Linear(16*18*3*3, 128),  #2592
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        
            nn.Linear(64, n_outputs))
        
    def forward(self, x):
        x = self.features(x)
        shape = list(x.size())
        batch_size = int(shape[0]/140)
        x = x.view(batch_size, 140, 512, 10, 10)
        x = self.reducingconvs(x)
        #output is shape torch.Size([1, 16, 18, 3, 3])
        x = x.view(batch_size, 16*18*3*3)
        x = self.classifier(x)
        return x

class ResNet152(nn.Module):
    def __init__(self, n_outputs, bottleneck, bottleneck_size):
        super(ResNet152, self).__init__()
        #Keep the layers of the resnet closest to the input:
        resnet = models.resnet152(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-4]))
        
        #Fix the weights in the feature extractor: 
        for param in self.features.parameters():
            param.requires_grad = False
        
        #conv input [1, 140, 512, 38, 38] on PACE
        # torch.nn.Conv2d or Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(140, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (2,2,2), stride=(2,2,2), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(16, 16, kernel_size = (2,2,2), stride=(2,2,2), padding=0),
            nn.ReLU(inplace=True))
        
        self.classifier = nn.Sequential(
            nn.Linear(16*14*8*8, 128), #14,336
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))
        
        if bottleneck:
            self.classifier2 = nn.Sequential(
                 nn.Linear(64, bottleneck_size), #Bottleneck down to bottleneck_size, e.g. 16
                 nn.ReLU(inplace=True),
                 nn.Dropout(0.5),
                 
                 nn.Linear(bottleneck_size, n_outputs))
        else:
            self.classifier2 = nn.Sequential(
                nn.Linear(64, n_outputs))
        
    def forward(self, x):
        x = self.features(x)
        shape = list(x.size())
        batch_size = int(shape[0]/140)
        #torch.Size([140, 512, 38, 38])
        x = x.view(batch_size, 140, 512, 38, 38)
        x = self.reducingconvs(x)
        #output is shape torch.Size([1, 16, 14, 8, 8])
        x = x.view(batch_size, 16*14*8*8)
        x = self.classifier(x)
        x = self.classifier2(x)
        return x

