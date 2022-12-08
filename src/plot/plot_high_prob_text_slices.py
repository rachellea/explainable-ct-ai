#plot_high_prob_text_slices.py
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

import os
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ioff()

def make_plot_of_high_prob_text_slices(results_dir, epoch, ct_scan_path,
                                       lower_thresh, upper_thresh):
    """Make a plot of the CT slices that have probability
    lower_thresh <= p <= upper_thresh
    of including text, according to the results of deploying a text
    classification model on the data set."""
    #Example results dir: '/home/rlb61/data/img-hiermodel2/results/results_2021/2021-02-11_TextClassifier2_Cube_FullAug_DeployEp56OnValid_axial/'
    #The results dir should be the results dir for a text classification model
    #that was deployed on uncropped unpadded scans
    assert os.path.isdir(results_dir)
    
    subdir_name = 'top_slice_plots_'+str(round(lower_thresh*100))+'_to_'+str(round(upper_thresh*100))
    if not os.path.exists(os.path.join(results_dir,subdir_name)):
        os.mkdir(os.path.join(results_dir,subdir_name))
    
    #infer the setname
    setname = infer_setname_from_results_dir(results_dir)
    
    #Read in the predicted probabilities
    #index is volume_acc, columns are slices, values are probabilities
    #example volume acc: 'val2345.npz'
    #example slice: 'slice_298'
    pred_probs = pd.read_csv(os.path.join(os.path.join(results_dir,'pred_probs'), setname+'_predprob_ep'+str(epoch)+'.csv'),
                        header=0,index_col=0)
        
    #First, figure out which scans have ANY slice with lower_thresh <= p
    max_over_cols = pred_probs.max(axis=1) #index is volume_acc, value is max probability across all slices
    gr_than_lower = max_over_cols[max_over_cols>=lower_thresh] #same as max_over_cols but now only scans with a max p >= lower_thresh are included
    
    #Iterate through all those scans and visualize the "guilty slices"
    for volume_acc in gr_than_lower.index.values.tolist():
        print('Making top text slices plot for',volume_acc)
        
        #Load CT volume and transpose if needed
        ctvol = np.load(os.path.join(ct_scan_path, volume_acc))['ct']
        if 'coronal' in results_dir: #[cor, sag, ax]
            ctvol = np.transpose(ctvol,[1,0,2])
        elif 'sagittal' in results_dir: #[sag, ax, cor]
            ctvol = np.transpose(ctvol,[2,0,1])
        
        #how many guilty slices for this ct?
        pred_probs_this_volume = pred_probs.loc[volume_acc,:]
        this_volume_slices = pred_probs_this_volume[(pred_probs_this_volume >= lower_thresh) & (pred_probs_this_volume <=upper_thresh)]
        num_this_volume_slices = this_volume_slices.shape[0]
        
        #determine the number of subplots
        #if num_rows != num_cols, then num_cols will always be bigger
        num_rows = math.floor(math.sqrt(num_this_volume_slices))
        num_cols = math.ceil(num_this_volume_slices/num_rows)
        
        #init plot
        fig, ax = plt.subplots(num_rows, num_cols,
                               figsize=(3*num_cols,3*num_rows))
        
        #fill in plot
        plotting_slice_idx = 0
        for row in range(num_rows):
            for col in range(num_cols):
                if plotting_slice_idx < num_this_volume_slices:
                    slice_name = this_volume_slices.index.values.tolist()[plotting_slice_idx] #e.g. 'slice_203'
                    slice_prob = this_volume_slices[slice_name]
                    assert slice_prob >= lower_thresh
                    assert slice_prob <= upper_thresh
                    slicenum = int(slice_name.replace('slice_',''))
                    
                    #plot the slice
                    slice_pixels = ctvol[slicenum,:,:]
                    if ((num_cols==1) and (num_rows==1)):
                        #no indexing is needed
                        ax.imshow(slice_pixels, cmap = plt.cm.gray)
                        ax.set_title(str(slicenum)+', p='+str(round(slice_prob,3)))
                        ax.set_xticks([])
                        ax.set_yticks([])
                    elif num_rows==1:
                        #if num_rows!=num_cols then num_cols will always be
                        #bigger. If num_rows is 1, then we must index by
                        #just col.
                        ax[col].imshow(slice_pixels, cmap = plt.cm.gray)
                        ax[col].set_title(str(slicenum)+', p='+str(round(slice_prob,3)))
                        ax[col].set_xticks([])
                        ax[col].set_yticks([])
                    else: #bow num_rows and num_cols are greater than 1
                        ax[row,col].imshow(slice_pixels, cmap = plt.cm.gray)
                        ax[row,col].set_title(str(slicenum)+', p='+str(round(slice_prob,3)))
                        ax[row,col].set_xticks([])
                        ax[row,col].set_yticks([])
                    
                    #increment which CT slice we're selecting
                    plotting_slice_idx+=1
                else:
                    #we're out of CT slices to plot, so hide the relevant subplots
                    ax[row,col].axis('off')
        
        plt.suptitle(volume_acc,fontsize='xx-large')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(os.path.join(results_dir,subdir_name),volume_acc.replace('.npz','_text_slices.png')))
        plt.close()
    
def plot_prob_histogram(results_dir, epoch):
    #Histogram of predicted probabilities
    setname = infer_setname_from_results_dir(results_dir)
    view = infer_view_from_results_dir(results_dir)
    pred_probs = pd.read_csv(os.path.join(os.path.join(results_dir,'pred_probs'),setname+'_predprob_ep'+str(epoch)+'.csv'),
                        header=0,index_col=0)
    plt.figure(figsize=(10, 7.5))  
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}')) #format thousands with commas on y-axis
    ax.spines['top'].set_visible(False)  
    ax.spines['right'].set_visible(False)  
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()  
    plt.xticks(fontsize=16)  
    plt.yticks(fontsize=16)
    plt.xlabel('Text Probability', fontsize=16)  
    plt.ylabel('Slice Count', fontsize=16)
    #Remove zeros from the data, since values of zero correspond to slices
    #that do not actually exist (and were just in the dataframe because the
    #dataframe had to include headers up to the max number of slices of any scan)
    data_nonzero = [x for x in pred_probs.values.flatten().tolist() if x != 0]
    plt.hist(data_nonzero, bins=40, rwidth = 0.7, color='#3F5D7D')
    plt.title(view.capitalize()+' '+setname.capitalize()+' Set Per-Slice Text Probability',fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,view.capitalize()+'_'+setname.capitalize()+'_Set_Histogram_of_Predicted_Text_Probabilities.png'))
    plt.close()

def infer_setname_from_results_dir(results_dir):
    if 'valid' in results_dir.lower():
        return 'valid'
    elif 'train' in results_dir.lower():
        return 'train'
    elif 'test' in results_dir.lower():
        return 'test'

def infer_view_from_results_dir(results_dir):
    if 'axial' in results_dir.lower():
        return 'axial'
    elif 'coronal' in results_dir.lower():
        return 'coronal'
    elif 'sagittal' in results_dir.lower():
        return 'sagittal'