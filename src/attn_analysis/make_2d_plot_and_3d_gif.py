#make_2d_plot_and_3d_gif.py
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
import imageio
import numpy as np
import torch, torch.nn as nn

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ioff()

def plot_attn_over_ct_scan(ctvol, attn_volume, x_perslice_scores_this_disease,
                           volume_acc, label_name, attn_2dplot_dir, attn_3dgif_dir):
    """
    <ctvol> is a torch Tensor with shape [1, 135, 3, 420, 420]
    <attn_volume> is a np array with shape [135, 6, 6]
        Note that attn_volume is the segprediction_clipped_and_normed from
        run_attn_analysis.py and thus it has already had a ReLU and
        normalization applied.
    <x_perslice_scores_this_disease> is a np array with the raw scores for
        all the slices for this particular disease
    <volume_acc> is a string e.g. 'RHAA12345_6.npz'
    <label_name> is a string e.g. 'lung_atelectasis'
    <attn_2dplot_dir> and <attn_3dgif_dif> are strings that are paths to
        directories where the results should be stored."""
    #Reshape the ctvol
    ctvol = ctvol.squeeze(dim=0) #out shape [135,3,420,420]
    ctvol = ctvol.data.cpu().numpy()
    ctvol = np.reshape(ctvol, newshape=[ctvol.shape[0]*ctvol.shape[1], ctvol.shape[2], ctvol.shape[3]]) #out shape [405, 420, 420]
        
    #Upsample the attention
    attn_volume = upsample_volume(attn_volume, out_shape=ctvol.shape)
    
    #2d plot
    savepath = os.path.join(attn_2dplot_dir, volume_acc.replace('.npz','')+'_'+label_name)
    sorted_indices = np.argsort(-x_perslice_scores_this_disease).flatten() #e.g. array([ 46,  31,  37, ...,118, 132, 117]) The indices that produce a sorted array
    #Check:
    #(Pdb) x_perslice_scores_this_disease[0,sorted_indices]
    # array([ 2.64493923e+01,  2.40455704e+01,  2.27083874e+01,  2.23833179e+01,
    #         2.22942944e+01,  2.21629314e+01,  2.20393772e+01,  2.20012302e+01,
    #         2.19151287e+01,  2.13861923e+01,  2.13474636e+01,  2.11356544e+01, etc.
    top_five_slice_idxs = sorted_indices[0:5] #the indices of the slices with the top 5 highest scores, e.g. array([46, 31, 37, 47, 38])
    #Check:
    #(Pdb) x_perslice_scores_this_disease[0,top_five_slice_idxs]
    #array([26.449392, 24.04557 , 22.708387, 22.383318, 22.294294],dtype=float32)
    
    for list_position in range(len(top_five_slice_idxs)): #list_position is 0, 1, 2, 3, 4
        slice_idx = top_five_slice_idxs[list_position] #e.g. slice_idx is 46 for list_position 0
        rank = list_position+1 #rank 1, 2, 3, 4, or 5 (top 5 slices)
        slice_score = round(x_perslice_scores_this_disease[0,slice_idx],2) #e.g. 26.45
        make_2d_plot_top_slice(attn_volume, ctvol, slice_idx, slice_score, rank, savepath, label_name)
    
    ##3d gif
    #savepath = os.path.join(attn_3dgif_dir, volume_acc.replace('.npz','')+'_'+label_name)
    #make_whole_gif(attn_volume, ctvol, savepath, label_name)

###########
# 2D Plot #---------------------------------------------------------------------
###########
def make_2d_plot_top_slice(attn_volume, ctvol, slice_idx, slice_score, rank,
                           savepath, label_name):
    """Make a 2D plot of the top slice for this disease with the attention
    heatmap overlaid.
    <attn_volume> is a np array with shape [405, 420, 420]
    <ctvol> is a np array with shape [405, 420, 420]
    <slice_idx> is an integer, for the index of the top slice. Note that this
        index is assuming we have a 3-channel image instead of a 1-channel
        image, so this slice_idx will have to be multiplied by 3.
    <slice_score> is the numerical value of the slice score
    <rank> is an int indicating which rank slice this is. if rank = 3 then this
        is the slice with the 3rd highest score for this disease
    <savepath> is a string; the path to save the output minus the extension
    <label_name> is a string for the abnormality e.g. 'lung_mass' """
    ctvol_square = ctvol[slice_idx*3,:,:] #out shape [420, 420]
    attn_square = attn_volume[slice_idx*3,:,:] #out shape [420, 420]
    fig, ax = plt.subplots(figsize=(24,18))
    plt.imshow(ctvol_square,cmap='gray')
    plt.imshow(attn_square, cmap='rainbow', alpha=0.5)
    #plt.title(label_name.replace('_',' ')+' rank '+str(rank)+' with score '+str(slice_score))
    #Turn off all axis markings:
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    plt.savefig(savepath+'rank'+str(rank)+'_2dplot.png', bbox_inches='tight')
    plt.close()

##########
# 3D GIF #----------------------------------------------------------------------
##########
def make_whole_gif(attn_volume, ctvol, savepath, label_name):
    """Make a gif of the entire CT scan with the attention heatmap overlaid.
    <attn_volume> is a np array with shape [405, 420, 420]
    <ctvol> is a np array with shape [405, 420, 420]
    <savepath> is a string; the path to save the output minus the extension"""
    #Use a matplotlib.colors.Normalize setup so that the colormap is normalized
    #consistently across all of the slices (otherwise the min and max of each
    #slice separately will be used as the min and max of the colormap, which
    #will destroy heatmap color meaning between slices)
    norm = matplotlib.colors.Normalize(vmin=np.amin(attn_volume),vmax=np.amax(attn_volume),clip=False)
    
    #Now collect the images for the gif
    images = [] #this will hold the images to make our gif
    for slice_idx in range(ctvol.shape[0]):
        ctvol_square = ctvol[slice_idx,:,:] #out shape [420, 420] = [square, square]
        attn_square = attn_volume[slice_idx,:,:] #out shape [420, 420]
        
        #https://ndres.me/post/matplotlib-animated-gifs-easily/
        fig, ax = plt.subplots(figsize=(8,6))
        plt.imshow(ctvol_square,cmap='gray')
        plt.imshow(attn_square, cmap='rainbow', alpha=0.5, norm = norm)
        plt.title(label_name.replace('_',' '))
        #Turn off all axis markings:
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        fig.canvas.draw() #draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close()
    imageio.mimsave(savepath+'_whole.gif',images,duration=0.5)

def upsample_volume(volume, out_shape):
    upsampled_tensor = nn.functional.interpolate(torch.Tensor(volume).unsqueeze(0).unsqueeze(0), size=out_shape, mode='trilinear').squeeze() #out shape [405,420,420]
    #upsampled_tensor = nn.functional.interpolate(torch.Tensor(volume).unsqueeze(0).unsqueeze(0), size=out_shape, mode='nearest').squeeze() #out shape [405,420,420]
    return upsampled_tensor.numpy() #out shape [405,420,420]
    