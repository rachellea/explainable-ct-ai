#custom_models_projections.py
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

class BodyAvg_NoProjection(nn.Module):
    """BodyAvg (baseline) model with no projection
    This is a separate class because it needs to load 3 channel images to
    avoid memory errors.
       (1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs):
        super(BodyAvg_NoProjection, self).__init__()
        self.slices_init = 405 #loading from raw
        self.slices_after_projection_and_3_channels = 135 #note there is NO PROJECTION
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices_after_projection_and_3_channels)
        
    def forward(self, x):
        #Check raw data shape
        #Note: UNLIKE BodyAvg_Projected, this model expects three channel
        #input to start, e.g. [1, 135, 3, 420, 420]
        shape = list(x.size())
        batch_size = int(shape[0])
        slices = int(shape[1])
        channels = int(shape[2])
        assert int(shape[3])==int(shape[4])
        square = int(shape[3])
        assert batch_size == 1
        assert slices==self.slices_after_projection_and_3_channels
        assert slices*3 == self.slices_init
        assert channels == 3
        assert square==420
        
        #Model
        x = x.squeeze(0)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        x = self.fc(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class BodyAvg_Projected(nn.Module):
    """BodyAvg (baseline) model with max pooling projections.
       (1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, projection):
        super(BodyAvg_Projected, self).__init__()
        assert projection in [3,9,27]
        if projection==3:
            self.slices_init = 405 #loading from raw
            self.do_max_pool = True
            n_pool = 3 #max pooling across 3 slices since projection==3
            self.slices_after_projection = 135
            self.slices_after_projection_and_3_channels = 45
        elif projection==9:
            self.slices_init = 45 #loading from 9 projections
            self.do_max_pool = False
            self.slices_after_projection = 45
            self.slices_after_projection_and_3_channels = 15
        elif projection==27:
            self.slices_init = 45 #loading from 9 projections
            self.do_max_pool = True
            n_pool = 3 #max pooling across 3 slices since each 'slice' is 9 already
            self.slices_after_projection = 15
            self.slices_after_projection_and_3_channels = 5
        
        assert self.slices_after_projection/3 == self.slices_after_projection_and_3_channels
        
        if self.do_max_pool:
            self.pooling = nn.MaxPool3d(kernel_size = (n_pool,1,1), stride=(n_pool,1,1), padding=0)
        
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices_after_projection_and_3_channels)
        
    def forward(self, x):
        #Check raw data shape
        #note: this model expects one-channel input to start, e.g. [1, 405, 420, 420]
        shape = list(x.size())
        batch_size = int(shape[0])
        slices = int(shape[1])
        square = shape[3]
        assert batch_size == 1
        assert slices==self.slices_init
        assert square==420
        
        #Do projection if applicable
        x = x.unsqueeze(0)
        if self.do_max_pool:
            x = self.pooling(x) #e.g. for projection=3, out [1, 1, 135, 420, 420]
        x = x.squeeze() #e.g. out [135, 420, 420]
        assert list(x.size())[0]==self.slices_after_projection
        assert list(x.size())[1]==list(x.size())[2]==square==420
        
        #Channelify
        x = x.view(self.slices_after_projection_and_3_channels,3,square,square) #TODO double check this
        
        #Model
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        x = self.fc(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class BodyAvg_Projected_RANDPADBEFORE(nn.Module):
    """BodyAvg (baseline) model with max pooling projections.
    I'm rerunning the 9 and 27 projections for max pooling because I realized
    that the random padding has a totally different effect if it's applied
    before projection (what this model will accomplish) versus if it's applied
    after projection (what I ended up running previously.)
       (1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, projection):
        super(BodyAvg_Projected_RANDPADBEFORE, self).__init__()
        print('BodyAvg_Projected_RANDPADBEFORE')
        assert projection in [9,27]
        self.slices_init = 405 #loading from raw for all of these
        if projection==9:
            self.slices_after_projection = 45
            self.slices_after_projection_and_3_channels = 15
        elif projection==27:
            self.slices_after_projection = 15
            self.slices_after_projection_and_3_channels = 5
        
        assert self.slices_after_projection/3 == self.slices_after_projection_and_3_channels
        
        #Average pooling projections
        self.pooling = nn.MaxPool3d(kernel_size = (projection,1,1), stride=(projection,1,1), padding=0)
        
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices_after_projection_and_3_channels)
        
    def forward(self, x):
        #Check raw data shape
        #note: this model expects one-channel input to start, e.g. [1, 405, 420, 420]
        shape = list(x.size())
        batch_size = int(shape[0])
        slices = int(shape[1])
        square = shape[3]
        assert batch_size == 1
        assert slices==self.slices_init
        assert square==420
        
        #Do projection
        x = x.unsqueeze(0)
        x = self.pooling(x) #e.g. for projection=3, out [1, 1, 135, 420, 420]
        x = x.squeeze() #e.g. out [135, 420, 420]
        assert list(x.size())[0]==self.slices_after_projection
        assert list(x.size())[1]==list(x.size())[2]==square==420
        
        #Channelify
        x = x.view(self.slices_after_projection_and_3_channels,3,square,square) #TODO double check this
        
        #Model
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        x = self.fc(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

###################
# Average Pooling #-------------------------------------------------------------
###################
class BodyAvg_AverageProjected(nn.Module):
    """BodyAvg (baseline) model with average pooling projections.
    Note that I don't have any precomputed average pooled CT scans so I need to
    calculate average pooling projections for everything.
       (1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, projection):
        super(BodyAvg_AverageProjected, self).__init__()
        assert projection in [3,9,27]
        self.slices_init = 405 #loading from raw for all of these
        if projection==3:
            self.slices_after_projection = 135
            self.slices_after_projection_and_3_channels = 45
        elif projection==9:
            self.slices_after_projection = 45
            self.slices_after_projection_and_3_channels = 15
        elif projection==27:
            self.slices_after_projection = 15
            self.slices_after_projection_and_3_channels = 5
        
        assert self.slices_after_projection/3 == self.slices_after_projection_and_3_channels
        
        #Average pooling projections
        self.pooling = nn.AvgPool3d(kernel_size = (projection,1,1), stride=(projection,1,1), padding=0)
        
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices_after_projection_and_3_channels)
        
    def forward(self, x):
        #Check raw data shape
        #note: this model expects one-channel input to start, e.g. [1, 405, 420, 420]
        shape = list(x.size())
        batch_size = int(shape[0])
        slices = int(shape[1])
        square = shape[3]
        assert batch_size == 1
        assert slices==self.slices_init
        assert square==420
        
        #Do projection
        x = x.unsqueeze(0)
        x = self.pooling(x) #e.g. for projection=3, out [1, 1, 135, 420, 420]
        x = x.squeeze() #e.g. out [135, 420, 420]
        assert list(x.size())[0]==self.slices_after_projection
        assert list(x.size())[1]==list(x.size())[2]==square==420
        
        #Channelify
        x = x.view(self.slices_after_projection_and_3_channels,3,square,square) #TODO double check this
        
        #Model
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        x = self.fc(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

#############################
# Selecting Every Kth Slice #---------------------------------------------------
#############################
class BodyAvg_SelectEveryKthSlice(nn.Module):
    """BodyAvg (baseline) model where every k^th slice is selected for inclusion.
    average pooling projections.
    Note that I don't have any precomputed k^th slice selected CT scans so I have
    to select out the nth slices starting from the raw data for everything.
    <chosen_k> indicates the k value to use when selecting slices.
       (1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, chosen_k):
        super(BodyAvg_SelectEveryKthSlice, self).__init__()
        assert chosen_k in [3,9,27]
        self.chosen_k = chosen_k
        self.slices_init = 405 #loading from raw for all of these
        if chosen_k==3:
            self.slices_after_every_kth = 135
            self.slices_after_every_kth_and_3_channels = 45
        elif chosen_k==9:
            self.slices_after_every_kth = 45
            self.slices_after_every_kth_and_3_channels = 15
        elif chosen_k==27:
            self.slices_after_every_kth = 15
            self.slices_after_every_kth_and_3_channels = 5
        
        assert self.slices_after_every_kth/3 == self.slices_after_every_kth_and_3_channels
        
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices_after_every_kth_and_3_channels)
        
    def forward(self, x):
        #Check raw data shape
        #note: this model expects one-channel input to start, e.g. [1, 405, 420, 420]
        shape = list(x.size())
        batch_size = int(shape[0])
        slices = int(shape[1])
        square = shape[3]
        assert batch_size == 1
        assert slices==self.slices_init
        assert square==420
        
        #Do selection of every nth slice
        upper_z = int(self.slices_init/self.chosen_k)
        every_kth_index = [z*self.chosen_k for z in range(0,upper_z)]
        x = x[:,every_kth_index,:,:]
        x = x.squeeze(0) #get rid of batch dimension since slices will serve as batch for feature extractor
        assert list(x.size())[0]==self.slices_after_every_kth
        assert list(x.size())[1]==list(x.size())[2]==square==420
        
        #Channelify
        x = x.view(self.slices_after_every_kth_and_3_channels,3,square,square) #TODO double check this
        
        #Model
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        x = self.fc(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x