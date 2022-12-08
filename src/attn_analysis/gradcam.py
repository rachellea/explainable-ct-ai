#gradcam.py
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

import os
import cv2
import copy
import torch
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ioff()

from . import model_outputs_classes

class RunGradCAM:
    def __init__(self, attention_type, model, device, label_meanings, results_dir,
                 task, model_name, target_layer_name):
        """TODO ADD DOCUMENTATION
        
        This class was based on code in jacobgil/pytorch-grad-cam
        https://github.com/jacobgil/pytorch-grad-cam/blob/master/grad-cam.py"""
        self.attention_type = attention_type
        self.model = model
        self.model.eval()
        self.modeloutputsclass = model_outputs_classes.return_modeloutputsclass(model_name)
        self.device = device
        self.label_meanings = label_meanings #all the diseases IN ORDER
        self.results_dir = results_dir
        self.task = task
        self.model_name = model_name
        self.target_layer_name = target_layer_name #e.g. '2'
    
    def return_segprediction_from_grad_cam(self, ctvol, gr_truth, volume_acc,
                                           chosen_label_index):
        """Do Grad-CAM on a CT volume <ctvol>. The ground truth labels for
        this ctvol are provided in <gr_truth>. The name of the ctvol is
        provided as a string in <volume_acc>.
            ctvol is a torch Tensor with shape [1, 135, 3, 420, 420]; in
                model_outputs_classes.py the batch dimension of 1 gets removed
                before putting it through the model.
            gr_truth.shape = TODO
            chosen_label_index is an integer
        """
        #obtain gradients and activations:
        extractor = self.modeloutputsclass(self.model, self.target_layer_name)
        self.all_target_activs_dict, x_perslice_scores, output = extractor.run_model(ctvol)
        
        #Use <one_hot> to multiply by zero every score except the score
        #for the target disease:
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][chosen_label_index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(self.device)
        one_hot = torch.sum(one_hot * output)
        
        #Backward pass:
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        #grads_list is a list of gradients, for each of the target layers.
        #Hooks are registered when we do the backward pass, which is why
        #we needed to wait until after calling backward() to get the
        #gradients.
        self.all_grads_dict = extractor.get_gradients()
        
        #Select gradients and activations for the target layer:
        target_grads = self.all_grads_dict[self.target_layer_name].cpu().data.numpy() #e.g. out shape [135, 16, 6, 6]
        target_activs = self.all_target_activs_dict[self.target_layer_name].cpu().data.numpy() #e.g. out shape [135, 16, 6, 6]
        
        #Figure out the slice_idx of the slice with the highest score for the
        #chosen disease:
        if self.model_name in ['AxialNet','AxialNet_Mask']:
            x_perslice_scores_this_disease = x_perslice_scores[:,chosen_label_index,:].cpu().data.numpy()
            #slice_idx is an int, e.g. 13; the index of
            #the slice with the highest score for the selected disease
            #Note that argsort by default spits out the indices so that the
            #data will be arranged from smallest to largest. Here I am sorting
            #the 'negative score', which means that the index in
            #the 0th position will correspond to the BIGGEST positive score,
            #which is the one I want.
            slice_idx = (np.argsort(-x_perslice_scores_this_disease).flatten()[0])
        elif self.model_name == 'CTNetModel':
            #we don't have scores for each slice because this isn't a MIL model
            #So instead, let's pick the slice with the highest gradient.
            #The target_grads for this model have shape [135, 512, 14, 14]
            grad_maxes = np.amax(target_grads,axis=(1,2,3)) #out shape [135]
            #The last item is biggest
            slice_idx = np.argsort(grad_maxes)[-1]
        else:
            x_perslice_scores_this_disease = None #for compatibility
            
        # if 'attn_plots' in self.task:
        #     #the anon_savename is 'anon' because it does not include the volume
        #     #accession and therefore it can be used in the figure title without
        #     #revealing any PHI
        #     anon_savename = self.label_meanings[chosen_label_index]+'_layer'+self.target_layer_name+'_slice'+str(slice_idx)
        #     savepath = os.path.join(self.results_dir, volume_acc.replace('.npz','')+anon_savename)
        #     #make this 2d stepbystep figure here:
        #     make_slice_2d_stepbystep_plot(self.attention_type, target_activs, target_grads,
        #         ctvol, chosen_label_index, slice_idx,
        #         anon_savename, savepath)
        if self.attention_type == 'gradcam-vanilla':
            return gradcam_vanilla(target_grads, target_activs), x_perslice_scores_this_disease
        elif self.attention_type == 'hirescam':
            return hirescam(target_grads, target_activs), x_perslice_scores_this_disease

def gradcam_vanilla(target_grads, target_activs):
    """Calculate vanilla GradCAM attention volume.
    An alpha_k is produced by taking the average of the gradients going in
    to the k^th feature map. The alpha_k is multipled against that feature map.
    The final Grad-CAM attention is the result of summing all the
    alpha_k*feature_map arrays.
    
    <target_grads> is a np array with shape [135, 16, 6, 6] which is
        height, features, square, square.
    <target_activs> is a np array [135, 16, 6, 6]"""
    target_grads_reshaped = np.transpose(target_grads,axes=(1,0,2,3)) #out shape [16, 135, 6, 6]
    alpha_ks = np.mean(target_grads_reshaped,axis=(1,2,3)) #out shape [16]
    alpha_ks_unsq = np.expand_dims(np.expand_dims(np.expand_dims(alpha_ks,axis=0),axis=2),axis=3) #out shape [1,16,1,1]
    product = np.multiply(target_activs,alpha_ks_unsq) #out shape [135, 16, 6, 6] from broadcasting
    raw_cam_volume = np.sum(product,axis=1) #out shape [135, 6, 6]
    
    #Note that the raw_cam_volume has not yet been ReLU'd or normalized
    return raw_cam_volume

def hirescam(target_grads, target_activs):
    """Calculate new proposed GradCAM attention volume.
    Here, the gradients going in to the k^th feature map are element-wise
    multiplied against the k^th feature map, and then the average is taken
    over the feature dimension."""
    #Improved step: get the 'CAM' by just doing element-wise multiplication of
    #the target_grads and the target_activs!!! and THEN collapsing across the
    #feature dimension. It's SO important to do the summing over the
    #feature dimension AFTER you have multiplied the grads and the activations.
    #Otherwise you are 'blurring' your result for no reason.
    raw_cam_volume = np.multiply(target_grads,target_activs) #e.g. out shape [135, 16, 6, 6]
    #Now sum over the feature dimension:
    raw_cam_volume = np.sum(raw_cam_volume,axis=1) #e.g. out shape [135, 6, 6]
    return raw_cam_volume

def _select_slice(target_activs, target_grads, slice_idx):
    """Return target_activs_slice and alpha_ks_slice which are the
    activation maps and alpha_ks for the given slice, respectively.
    
    <target_activs> is a np array with shape [135, 16, 6, 6]
    <target_grads> is a np array with shape [135, 16, 6, 6]"""
    target_activs_slice = target_activs[slice_idx,:,:,:] #out shape [16, 6, 6]
    grad_slice = target_grads[slice_idx,:,:,:] #out shape [16, 6, 6]
    alpha_ks_slice = np.mean(grad_slice, axis=(1,2)) #importance weights for the feature maps, e.g. out shape [16] mean across Height and Width
    return target_activs_slice, alpha_ks_slice

#TODO: REDO THIS FUNCTION SO YOU DO NOT HAVE ANYTHING 2D IN IT AT ALL!!!
#BECAUSE RIGHT NOW THE SELECT_SLICE STEP IS 2D WHICH IS WRONG
#YOU NEED TO DELETE THE SELECT_SLICE FUNCTION AND JUST REDO THE IMPLEMENTATION
#OF THIS WHOLE FUNCTION SO THAT IT'S RIGHT
#ALSO, FOLLOW THE CAPITALIZATION/NAMING SCHEME FOR THE PLOTS FROM ATTENTION REPO
def make_slice_2d_stepbystep_plot(attention_type, target_activs, target_grads,
                ctvol, chosen_label_index, slice_idx,
                anon_savename, savepath):
    """Make a 2D plot showing all of the intermediate calculations needed
    to create the Grad-CAM or HiResCAM heatmap for the slice with the highest
    score for the chosen disease.
    
    <attention_type> is a string either 'gradcam-vanilla' or 'hirescam'
    <target_activs> is np array with shape [135, 16, 6, 6]
    <target_grads> is a np array with shape [135, 16, 6, 6]
    <ctvol> is a torch Tensor with shape [1, 135, 3, 420, 420]
    <chosen_label_index> is an int for the index of the label being visualized
    <slice_idx> is an int for the index of the slice being visualized
    <anon_savename> and <savepath> are strings indicating where to save the plot"""
    #Initialize the figure
    num_features = target_activs.shape[1] #16
    fig, ax = plt.subplots(nrows = 4, ncols = num_features, figsize=(40,14))
    ax[1,0].set_ylabel('activ_map',fontsize='xx-large')
    ax[3,0].set_ylabel('cam',fontsize='xx-large')
    fig.suptitle(anon_savename, fontsize='xx-large')
    
    #the target_activs_slice are used for both kinds of GradCAM
    #target_activs_slice has shape [16,6,6]
    #alpha_ks_slice has shape [16,]
    target_activs_slice, alpha_ks_slice = _select_slice(target_activs, target_grads, slice_idx)
    norm_target_activs_slice = matplotlib.colors.Normalize(vmin=np.amin(target_activs_slice),vmax=np.amax(target_activs_slice),clip=False)
    
    #Make figure:
    if attention_type == 'gradcam-vanilla':
        ax[0,0].set_ylabel('alpha_k',fontsize='xx-large')
        ax[2,0].set_ylabel('alpha_k*activ_map',fontsize='xx-large')
        norm_alpha_ks_slice = matplotlib.colors.Normalize(vmin=np.amin(alpha_ks_slice),vmax=np.amax(alpha_ks_slice),clip=False)
        product = np.expand_dims(np.expand_dims(alpha_ks_slice,axis=1),axis=1)*target_activs_slice
        norm_product = matplotlib.colors.Normalize(vmin=np.amin(product), vmax=np.amax(product),clip=False)
        cam = np.zeros(target_activs_slice.shape[1:], dtype=np.float32)
        #in this for loop, idx is indexing over the feature dimension.
        #In this case there are 16 different feature maps.
        for idx, alpha_k in enumerate(alpha_ks_slice):
            if np.sign(alpha_k) == -1:
                title = '-'
            elif np.sign(alpha_k) == 1:
                title = '+'
            elif np.sign(alpha_k) == 0:
                title = '0'
            #numpy.full(shape, fill_value) returns a new array of given shape
            #and type, filled with fill_value. We'll use it to visualize the
            #alpha_ks:
            desired_shape = target_activs_slice[idx,:,:].shape
            ax[0,idx].imshow(np.full(desired_shape,alpha_k), cmap='rainbow', norm = norm_alpha_ks_slice)
            ax[0,idx].set_title(title,fontsize='xx-large') #set title indicating the sign of each alpha_k (helps with understanding)
            #Show the target_activs_slice
            ax[1,idx].imshow(target_activs_slice[idx, :, :], cmap='rainbow', norm = norm_target_activs_slice)
            #Calculate the component of the cam and visualize it:
            cam_component = alpha_k*target_activs_slice[idx, :, :]
            ax[2,idx].imshow(cam_component, cmap='rainbow', norm = norm_product)
            cam += cam_component
    
    elif attention_type == 'hirescam':
        ax[0,0].set_ylabel('gradients',fontsize='xx-large')
        ax[2,0].set_ylabel('grad*activ_map',fontsize='xx-large')
        #target_grads.shape [135, 16, 6, 6]
        target_grads_slice = target_grads[slice_idx,:,:,:] #out shape [16,6,6]
        norm_target_grads_slice = matplotlib.colors.Normalize(vmin=np.amin(target_grads_slice),vmax=np.amax(target_grads_slice),clip=False)
        cam_slice = np.multiply(target_activs[slice_idx,:,:,:],target_grads[slice_idx,:,:,:]) #out shape [16,6,6]
        norm_cam_slice = matplotlib.colors.Normalize(vmin=np.amin(cam_slice), vmax=np.amax(cam_slice),clip=False)
        for idx in range(target_grads.shape[1]): #e.g. across 32 features
            ax[0,idx].imshow(target_grads_slice[idx, :, :], cmap='rainbow', norm = norm_target_grads_slice)
            ax[1,idx].imshow(target_activs_slice[idx, :, :], cmap='rainbow', norm = norm_target_activs_slice)
            ax[2,idx].imshow(cam_slice[idx, :, :], cmap='rainbow', norm = norm_cam_slice)
        cam = np.mean(cam_slice,axis=0) #mean across the features. out shape [6,6]
    
    #Show raw CAM in first slot of third row
    ax[3,0].imshow(copy.deepcopy(cam), cmap='rainbow')
    ax[3,0].set_title('raw',fontsize='xx-large')
    
    #Show rescaled cam in next slot
    cam = np.maximum(cam, 0) #ReLU operation
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    ax[3,1].imshow(copy.deepcopy(cam), cmap='rainbow')
    ax[3,1].set_title('rescaled',fontsize='xx-large')
    
    #Show CAM upscaled in next slot of third row
    cam_upscale = cv2.resize(cam, dsize=(420,420), interpolation=cv2.INTER_CUBIC)
    ax[3,2].imshow(cam_upscale, cmap='rainbow')
    ax[3,2].set_title('upsampled',fontsize='xx-large')
    
    #Finally show CAM and the middle channel of the relevant slice:
    #Select middle channel of the ctvol slice:
    ctvol_slice = ctvol.squeeze(dim=0)[slice_idx,:,:,:].cpu().data.numpy() #e.g. out shape [3,420,420] = [channels, square, square]
    ctvol_square = ctvol_slice[1,:,:] #out shape [420, 420]
    ax[3,3].imshow(ctvol_square,cmap='gray')
    ax[3,3].imshow(cam_upscale, cmap='rainbow', alpha=0.5)
    ax[3,3].set_title('superimposed',fontsize='xx-large')
    
    #get rid of un-used axes in the third row:
    for idx in range(4,num_features):
        fig.delaxes(ax[3,idx])
    
    plt.savefig(savepath+'_StepByStep.png')
    plt.close()
    