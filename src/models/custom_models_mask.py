#custom_models_mask.py
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
import math #needed for calculation of weight and bias initialization
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
from . import custom_models_diseasereps as cmdr

class AxialNet_Mask(nn.Module):
    """Identical implementation to the one in custom_models_base.py except
    that it returns an intermediate calculation of the convolution step
    which will be used in calculating a mask-related loss.
    
       (1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, slices):
        super(AxialNet_Mask, self).__init__()
        self.slices = slices #equal to 15 for 9 projections
        self.n_outputs = n_outputs
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        assert list(x.shape)==[1,self.slices,3,420,420]
        x1 = x.squeeze() #out shape [slices,3,420,420]
        x1 = self.features(x1) #out shape [slices,512,14,14]
        x1f = self.conv2d(x1) #out shape [slices, 16, 6, 6]
        x2 = self.fc(x1f) #out shape [slices,n_outputs,1,1]
        x2 = torch.squeeze(x2) #out shape [slices, n_outputs]
        x2_perslice_scores = x2.transpose(0,1).unsqueeze(0) #out shape [1, n_outputs, slices]
        x2 = self.avgpool_1d(x2_perslice_scores) #out shape [1, n_outputs, 1]
        x2f = torch.squeeze(x2, dim=2) #out shape [1, n_outputs]
        
        #Now calculate what the disease specific representation is in the
        #intermediate calculation of the fc layer.
        #First, make n_outputs copies of the slices x 16 x 6 x 6 representation:
        x1_repeated = x1f.repeat(self.n_outputs,1,1,1,1) #out shape [n_outputs, slices, 16, 6, 6]
        #Now select the fc_weights:
        fc_weights = self.fc.weight #shape [132, 16, 6, 6], where 132 is n_outputs
        fc_weights_unsq = fc_weights.unsqueeze(dim=1) #out shape [n_outputs, 1, 16, 6, 6]
        #Now multiply element wise. Broadcasting will occur.
        #we have [n_outputs, slices, 16, 6, 6] x [n_outputs, 1, 16, 6, 6]
        disease_reps = torch.mul(x1_repeated, fc_weights_unsq) #out shape [n_outputs, slices, 16, 6, 6]
        
        out = {'out':x2f,
               'x_perslice_scores':x2_perslice_scores,
               'disease_reps':disease_reps}
        return out

class AxialNet_Mask_VanillaGradCAM(nn.Module):
    """Identical implementation to the one in custom_models_base.py except
    that it returns an intermediate calculation of the convolution step
    which will be used in calculating a mask-related loss; this intermediate
    calculation is based on vanilla Grad-CAM.
    
       (1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) FC layer (implemented via conv) to [n_outputs, 1, 1]
       (4) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs, slices):
        super(AxialNet_Mask_VanillaGradCAM, self).__init__()
        self.slices = slices #equal to 15 for 9 projections
        self.n_outputs = n_outputs
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.fc = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        assert list(x.shape)==[1,self.slices,3,420,420]
        x1 = x.squeeze() #out shape [slices,3,420,420]
        x1 = self.features(x1) #out shape [slices,512,14,14]
        x1f = self.conv2d(x1) #out shape [slices, 16, 6, 6]
        x2 = self.fc(x1f) #out shape [slices,n_outputs,1,1]
        x2 = torch.squeeze(x2) #out shape [slices, n_outputs]
        x2_perslice_scores = x2.transpose(0,1).unsqueeze(0) #out shape [1, n_outputs, slices]
        x2 = self.avgpool_1d(x2_perslice_scores) #out shape [1, n_outputs, 1]
        x2f = torch.squeeze(x2, dim=2) #out shape [1, n_outputs]
        
        #Now calculate what the disease specific representation is in the
        #intermediate calculation of the fc layer.
        #First, make n_outputs copies of the slices x 16 x 6 x 6 representation:
        x1_repeated = x1f.repeat(self.n_outputs,1,1,1,1) #out shape [n_outputs, slices, 16, 6, 6]
        #Now select the fc_weights. These weights are also the gradients leaving
        #the last layer.
        fc_weights = self.fc.weight #shape [80, 16, 6, 6], where 80 is n_outputs
        
        #To calculate the alpha_ks, we need to take the mean across the height
        #and width so that we get one alpha_k per feature per disease:
        #(confirmed that this is the mean across the 6x6 in the gradcam code)
        alpha_ks = torch.mean(fc_weights,dim=(2,3)) #out shape [n_outputs, 16]
        alpha_ks_unsq = alpha_ks.unsqueeze(dim=1).unsqueeze(dim=3).unsqueeze(dim=3) #out shape [n_outputs, 1, 16, 1, 1]
        
        #Now multiply element wise. Broadcasting will occur.
        #we have [n_outputs, slices, 16, 6, 6] x [n_outputs, 1, 16, 1, 1]
        disease_reps = torch.mul(x1_repeated, alpha_ks_unsq) #out shape [n_outputs, slices, 16, 6, 6]
        
        #the summing over the feature dimension takes place in the loss
        #calculation
        
        out = {'out':x2f,
               'x_perslice_scores':x2_perslice_scores,
               'disease_reps':disease_reps}
        return out

class AxialNet_Mask_Final3DConv(nn.Module):
    """Identical implementation to the one in custom_models_base.py except
    that it returns an intermediate calculation of the convolution step
    which will be used in calculating a mask-related loss.
    
       (1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) Final FC layer implemented via 3D convolution to produce [n_outputs]"""
    def __init__(self, n_outputs, slices):
        super(AxialNet_Mask_Final3DConv, self).__init__()
        self.slices = slices #equal to 15 for 9 projections
        self.n_outputs = n_outputs
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv() #out shape [slices, 16, 6, 6]
        #Final step is 3D convolution!
        #Rep is first reshaped to [1, 16, slices, 6, 6]
        self.fc = nn.Conv3d(16, n_outputs, kernel_size=(self.slices,6,6), stride=(self.slices,6,6), padding=0)
            
    def forward(self, x):
        assert list(x.shape)==[1,self.slices,3,420,420]
        x1 = x.squeeze() #out shape [slices,3,420,420]
        x1 = self.features(x1) #out shape [slices,512,14,14]
        x1 = self.conv2d(x1) #out shape [slices, 16, 6, 6]
        
        #Reshape:
        x1f = x1.transpose(0,1).unsqueeze(0) #out shape [1, 16, slices, 6, 6]
        #Final classification
        x2 = self.fc(x1f) #out shape [1,n_outputs,1,1,1]
        x2f = x2.squeeze(dim=2).squeeze(dim=2).squeeze(dim=2) #out shape [1,n_outputs]
        
        #TODO TEST THIS (or at least make visualizations of disease_reps)
        #Now calculate what the disease specific representation is in the
        #intermediate calculation of the fc layer.
        #First, make n_outputs copies of the 16 x slices x 6 x 6 representation:
        x1_repeated = x1f.squeeze(dim=0).repeat(self.n_outputs,1,1,1,1) #out shape [n_outputs, 16, slices, 6, 6]
        #Now select the fc_weights:
        fc_weights = self.fc.weight #shape [n_outputs, 16, slices, 6, 6]
        assert x1_repeated.shape==fc_weights.shape
        #Now multiply element wise. Broadcasting will occur.
        #we have [n_outputs, 16, slices, 6, 6] x [n_outputs, 16, slices, 6, 6]
        disease_reps_orig = torch.mul(x1_repeated, fc_weights) #out shape [n_outputs, 16, slices, 6, 6]
        
        #But for the attention ground truth calculation we assume that the
        #disease_reps has shape [n_outputs, slices, 16, 6, 6], so transpose!
        disease_reps = disease_reps_orig.transpose(1,2) #out shape [n_outputs, slices, 16, 6, 6]
        
        out = {'out':x2f,
               'disease_reps':disease_reps}
        return out

class BodyLocationAttn3Mask(nn.Module): #7/2/2020, updated 7/7/2020, redone for mask 8/27/2020
    """Modification on 8/27 involves the shape of the attention calculated.
    Old version calculated [1,1,1,6,6 attention]. This version calculates
    [1,slices,1,6,6] attention (i.e. fully 3d spatially.)
    There is also a special loss associated with this model which requires the
    model to match the organ attention to ground truth organ masks.
    
    OLD DOCUMENTATION from model that this model was based on,
    BodyLocationAttn3 in custom_models_diseasereps.py:
    See AxialNetDiseaseFeatureAttn for more documentation including code comments.
    Difference from AxialNetDiseaseFeatureAttn: uses spatial attention instead of
    feature attention. Specifically there is right lung, heart, and left lung
    spatial attention. Also, instead of being fixed weights every time, the
    weights are learned based on using the center slices (since the center
    slices are most indicative of where the right lung, heart, and left
    lung are located.) So this is trainable soft self-attention."""
    def __init__(self, n_outputs_lung, n_outputs_heart):
        super(BodyLocationAttn3Mask, self).__init__()
        self.slices = 15 #9 projections
        self.n_outputs = (2*n_outputs_lung)+n_outputs_heart
        self.n_outputs_lung = n_outputs_lung
        self.n_outputs_heart = n_outputs_heart
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        
        #Calculate the spatial attention based on ALL the slices
        in_size = self.slices*16*6*6
        out_size = self.slices*6*6
        self.heart_attn_fc = nn.Sequential(nn.Linear(in_size, out_size),nn.Sigmoid())
        self.left_lung_attn_fc = nn.Sequential(nn.Linear(in_size, out_size),nn.Sigmoid())
        self.right_lung_attn_fc = nn.Sequential(nn.Linear(in_size, out_size),nn.Sigmoid())
        self.fclayers_weights, self.fclayers_biases = cmdr.init_stacked_fc_layers(total_independent_fc_layers = self.n_outputs, in_features = 16*6*6)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        
        #Calculate attention mask based on all slices
        #This attention mask is basically doing low-dimensional organ
        #segmentation. The nice thing about doing the segmentation this way
        #is that the model can still look at both lungs when predicting a
        #lung disease but it's forced to look MORE at the relevant lung.
        all_slices_flat = x.flatten().unsqueeze(dim=0) #out shape [1,8640]
        #The spatial maps must be able to be broadcast multiplied against
        #a Tensor of shape [slices, n_outputs_organ, 16, 6, 6]
        self.heart_spatial = self.heart_attn_fc(all_slices_flat).reshape(self.slices,1,1,6,6) #out shape [slices,1,1,6,6]
        self.left_lung_spatial = self.left_lung_attn_fc(all_slices_flat).reshape(self.slices,1,1,6,6) #out shape [slices,1,1,6,6]
        self.right_lung_spatial = self.right_lung_attn_fc(all_slices_flat).reshape(self.slices,1,1,6,6) #out shape [slices,1,1,6,6]
        
        #Repeat x
        x = x.repeat(self.n_outputs,1,1,1,1) #out shape [n_outputs, slices, 16, 6, 6]
        x = x.transpose(0,1) #out shape [slices, n_outputs, 16, 6, 6]
        
        #Apply the attention maps
        #Must follow ground truth label order, which is heart, left_lung, right_lung
        x_heart = torch.mul(x[:,0:self.n_outputs_heart,:,:,:],self.heart_spatial)
        x_left_lung = torch.mul(x[:,self.n_outputs_heart:self.n_outputs_heart+self.n_outputs_lung,:,:,:],self.left_lung_spatial)
        x_right_lung = torch.mul(x[:,-1*self.n_outputs_lung:,:,:,:],self.right_lung_spatial)
        x = torch.cat((x_heart,x_left_lung,x_right_lung),dim=1) #out shape [slices, n_outputs, 16, 6, 6]
        
        x = x.flatten(start_dim=2,end_dim=4) #out shape [slices, n_outputs, 16*6*6] = [slices, n_outputs, 576]
        slice_preds = cmdr.apply_disease_fc_layers(x, self.fclayers_weights, self.fclayers_biases)
        x = slice_preds.transpose(0,1).unsqueeze(0) #out shape [1, n_outputs, slices]
        x = self.avgpool_1d(x) #out shape [1, n_outputs, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, n_outputs]
        
        out = {'out':x,
               'heart_spatial':self.heart_spatial,
               'left_lung_spatial':self.left_lung_spatial,
               'right_lung_spatial':self.right_lung_spatial}
        return out

class BodyDiseaseSpatialAttn4Mask(nn.Module): #7/7/2020 #TODO test this #Updated 8/27/2020 for mask
    """In this model a 3D attention mask of shape [slices,6,6] is calculated for
    each disease, before the classification step.
    
    Note that this model is identical to BodyDiseaseSpatialAttn4 except for
    its usage:
    (a) custom loss function: in the loss, the location information is used to
        determine what locations the disease-specific attention is allowed to
        look at. e.g. if there is atelectasis only in the left lung then the
        attention for atelectasis for that scan should be only in the place
        demarcated as left lung in the segmentation ground truth.
        Furthermore, if there is NO atelectasis present, then the attention
        for atelectasis should all be zero.
        In order to calculate this custom loss, this model has to return
        the attention maps in addition to the predictions.
    (b) custom labels: this model is different from everything else I have
        been doing because it assumes that we just want to predict lung
        diseases generically and so it only makes
               n_outputs_lung+n_outputs_heart predictions, rather than
            (2*n_outputs_lung+n_outputs_heart) predictions.
    
    OLD DOCUMENTATION from model that this model was based on,
    BodyDiseaseSpatialAttn4 in custom_models_diseasereps.py 
    See AxialNetDiseaseFeatureAttn for more documentation including code comments.
    Difference from BodyLocationAttn3: while 4 also uses spatial
    attention (like 3), 4 does spatial attention per disease instead of per
    location."""
    def __init__(self, n_outputs_lung, n_outputs_heart):
        super(BodyDiseaseSpatialAttn4Mask, self).__init__()
        self.slices = 15 #9 projections
        #NOTE that here, we have only n_outputs_lung overall! We are not doing
        #separate predictions for the right and left lungs!
        self.n_outputs = n_outputs_lung+n_outputs_heart
        self.n_outputs_lung = n_outputs_lung
        self.n_outputs_heart = n_outputs_heart
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        
        #Calculate per-disease spatial attention based on ALL the slices
        #Repeated representation: [slices, n_outputs, 16, 6, 6]
        #Attention shape we want: [slices, n_outputs, 1, 6, 6]
        self.nonlinearity = nn.Sigmoid()
        #FC layers for calculating the disease-specific spatial attention
        #For each disease and each element of the 6x6 I learn a different FC layer:
        self.fcattns_weights, self.fcattns_biases = cmdr.init_stacked_fc_layers(total_independent_fc_layers = self.n_outputs*6*6, in_features = 16)
        #FC layers for calculating the final disease predictions
        self.fclayers_weights, self.fclayers_biases = cmdr.init_stacked_fc_layers(total_independent_fc_layers = self.n_outputs, in_features = 16*6*6)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        x = x.repeat(self.n_outputs,1,1,1,1) #out shape [n_outputs, slices, 16, 6, 6]
        x = x.transpose(0,1) #out shape [slices, n_outputs, 16, 6, 6]
        
        #Calculate the disease-specific spatial attention:
        attn_raw_list = []
        for slice_num in range(self.slices):
            slice_data = x[slice_num,:,:,:,:] #out shape [n_outputs, 16, 6, 6]
            slice_data = slice_data.flatten(start_dim=2,end_dim=3).transpose(1,2) #out shape [n_outputs, 6*6, 16]
            slice_data = slice_data.flatten(start_dim=0,end_dim=1) #out shape [n_outputs*6*6, 16]
            temp1 = torch.mul(slice_data,self.fcattns_weights) #out shape [n_outputs*6*6, 16]
            temp2 = torch.sum(temp1,dim=1) #out shape [n_outputs*6*6]
            temp3 = (temp2+self.fcattns_biases).unsqueeze(0) #out shape [n_outputs*6*6]
            attn_raw_list.append(temp3)
        attn_raw = torch.cat(attn_raw_list,dim=0) #out shape [slices, n_outputs*6*6]
        attn_raw = torch.reshape(attn_raw,(self.slices,self.n_outputs,6*6)) #out shape [slices, n_outputs, 6*6]
        attn = self.nonlinearity(attn_raw) #out shape [slices, n_outputs, 6*6]
        attn = torch.reshape(attn,(self.slices,self.n_outputs,6,6)).unsqueeze(2) #out shape [slices, n_outputs, 1, 6, 6]
        
        #Apply the attention
        x_times_attn = torch.mul(x, attn) #out shape [slices, n_outputs, 16, 6, 6]
        
        #Disease predictions
        x = x.flatten(start_dim=2,end_dim=4) #out shape [slices, n_outputs, 16*6*6] = [slices, n_outputs, 576]
        slice_preds = cmdr.apply_disease_fc_layers(x, self.fclayers_weights, self.fclayers_biases)
        x = slice_preds.transpose(0,1).unsqueeze(0) #out shape [1, n_outputs, slices]
        x = self.avgpool_1d(x) #out shape [1, n_outputs, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, n_outputs]
        out = {'out':x,
               'attn':attn} #attn out shape [slices, n_outputs, 1, 6, 6]
        return out

class BodyDiseaseSpatialAttn5Mask(nn.Module): #7/7/2020 #TODO test this
    #On the natural images dataset, this model had better performance
    #than model 4
    """Exactly the same as the BodyDiseaseSpatialAttn5 model except that
    this returns the attn so that it can be trained with a loss function that
    acts on the attn as well.
    
    OLD DOCUMENTATION from model that this model was based on,
    BodyDiseaseSpatialAttn5 in custom_models_diseasereps.py:
    See AxialNetDiseaseFeatureAttn for more documentation including code comments.
    Difference from BodyDiseaseSpatialAttn4: whereas 4 learns a different
    mapping of 16 features -> 1 spatial attn value for each element of the 6x6
    square, 5 uses a convolution layer such that the mapping of 16 -> 1 is
    the same for each element of the 6x6 square"""
    def __init__(self, n_outputs, nonlinearity):
        super(BodyDiseaseSpatialAttn5Mask, self).__init__()
        self.slices = 15 #9 projections
        self.n_outputs = n_outputs
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        #Calculate the spatial attention based on center slices
        if nonlinearity == 'softmax':
            self.nonlinearity = nn.Softmax(dim=2)
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        
        #Conv layer for calculating the disease-specific spatial attention
        #For each disease and each element of the 6x6 I learn a different FC layer:
        self.attn_conv = nn.Sequential(
            nn.Conv2d(16, self.n_outputs, kernel_size = (1,1), stride=(1,1), padding=0),
            self.nonlinearity)
        
        #FC layers for calculating the final disease predictions
        self.fclayers_weights, self.fclayers_biases = cmdr.init_stacked_fc_layers(total_independent_fc_layers = self.n_outputs, in_features = 16*6*6)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        
        #Calculate the disease-specific spatial attention:
        attn = self.attn_conv(x).unsqueeze(2) #out shape [slices, 83, 1, 6, 6]
        
        #Apply the attention
        x = x.repeat(self.n_outputs,1,1,1,1) #out shape [83, slices, 16, 6, 6]
        x = x.transpose(0,1) #out shape [slices, 83, 16, 6, 6]
        x_times_attn = torch.mul(x, attn) #out shape [slices, 83, 16, 6, 6]
        
        #Disease predictions
        x = x.flatten(start_dim=2,end_dim=4) #out shape [slices, 83, 16*6*6] = [slices, 83, 576]
        slice_preds = cmdr.apply_disease_fc_layers(x, self.fclayers_weights, self.fclayers_biases)
        x = slice_preds.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        
        out = {'out':x,
               'attn':attn} #attn out shape [slices, n_outputs, 1, 6, 6]
        return out