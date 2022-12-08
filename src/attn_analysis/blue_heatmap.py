#blue_heatmap.py
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
import copy
import torch, torch.nn as nn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ioff()
import seaborn

from . import renaming_abnormalities_new as rn

def visualize_slicediseases(pred, gr_truth, x_perslice_scores, volume_acc,
                            results_dir, label_meanings, blue_heatmap_baseline):
    """Make a heatmap figure showing the slice predictions for the
    different diseases. This is a visualization of <x_perslice_scores> along
    with the ground truth.
       pred is a torch Tensor with shape = [1, 80]
       gr_truth is a numpy array with shape [80]
       x_perslice_scores is a np array with shape = [1, 80, slices] where
           slices is e.g. 15 or 135
       blue_heatmap_baseline is a pandas dataframe that will be subtracted from
           the calculated dataframe before visualization."""
    #Linearly transform each column (each disease) into the range (0,1). Do not
    #use a sigmoid function because that is nonlinear and will mess up the
    #appearance I think.
    x_perslice_scores = np.transpose(np.squeeze(x_perslice_scores)) #out shape [slices, 80]
    lower_bounds = np.min(x_perslice_scores, axis=0)
    upper_bounds = np.max(x_perslice_scores, axis=0)
    x_perslice_scores = (x_perslice_scores - lower_bounds) / (upper_bounds - lower_bounds)
    
    #Apply a sigmoid function to pred, to get the final probabilities, and to
    #squish the predictions into (0,1) for better visualization:
    sigmoid = nn.Sigmoid()
    pred = sigmoid(pred).cpu().data.numpy()
    
    #Put data into dataframe
    slices = x_perslice_scores.shape[0]
    dataframe = pd.DataFrame(x_perslice_scores,
                             index=['slice '+str(x) for x in range(0,slices)],
                             columns=label_meanings)
    dataframe.loc['Pred',:] = np.squeeze(pred)
    dataframe.loc['Truth',:] = gr_truth
    if blue_heatmap_baseline is not None:
        dataframe = dataframe - blue_heatmap_baseline
    
    #Make plots    
    plot_all_organ_heatmaps(dataframe, volume_acc, results_dir)
    return dataframe

def plot_all_organ_heatmaps(dataframe, volume_acc, results_dir):
    gv_and_media_cols, gv_and_media_cols_rename, heart_cols, heart_cols_rename, lung_cols, lung_cols_rename = rn.return_renamers()
    plot_organ_heatmap(dataframe, gv_and_media_cols, gv_and_media_cols_rename, volume_acc, 'great_vessel_and_mediastinum', results_dir)
    plot_organ_heatmap(dataframe, heart_cols, heart_cols_rename, volume_acc, 'heart', results_dir)
    plot_organ_heatmap(dataframe, lung_cols, lung_cols_rename, volume_acc, 'lung', results_dir)

def plot_organ_heatmap(df, cols, cols_rename, volume_acc, organ_name, results_dir):
    df_organ = df.filter(items=cols,axis='columns').rename(columns=cols_rename)
    df_organ_new = copy.deepcopy(df_organ)
    c = np.amin(df_organ_new.values)
    d = np.amax(df_organ_new.values)
    plt.figure(figsize=(16, 32))
    heatmap = seaborn.heatmap(df_organ_new, cmap = 'Blues', square=True, center=0.5)
    #fix for mpl bug that cuts off the top/bottom of seaborn viz
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    outpath = os.path.join(results_dir,volume_acc.replace('.npz','')+'_'+organ_name+'_blue_heatmap.pdf')
    heatmap.get_figure().savefig(outpath,bbox_inches='tight')
    plt.close()

####################################################################
# Calculate the average heatmap across scans with no abnormalities #------------
####################################################################
def get_baseline(chosen_dataset, model, blue_heatmaps_dir):
    """Obtain the average heatmap across all the scans in this data set that
    are negative for all abnormalities.
    <chosen_dataset> is a PyTorch dataset e.g. as defined in run_attn_analysis.py
    <model> is a PyTorch model e.g. as defined in run_attn_analysis.py"""
    print('Getting baseline for blue heatmap')
    #Get results dir and check if baseline already made:
    results_dir = os.path.join(blue_heatmaps_dir,'baseline')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    if os.path.exists(os.path.join(results_dir,'avg_baseline_df.csv')):
        return pd.read_csv(os.path.join(results_dir,'avg_baseline_df.csv'),header=0,index_col=0)
    
    #If you haven't already made 'baseline_x_perslice_scores.npy' make it now:
    #First figure out which volume accessions have no abnormalities:
    #label_counts has note accession as index and total label count as value
    label_counts = chosen_dataset.labels_df.sum(axis=1)
    #List of note accs for CTs with no abnormalities: ['AA123','AA456','AA789']
    note_accs_no_abns = label_counts[label_counts==0].index.values.tolist()
    vol_accs_no_abns = chosen_dataset.volume_log_df.filter(items=note_accs_no_abns,axis='index')['full_filename_npz'].values.tolist()
    
    #Now get the indices of these scans
    chosen_dataset_indices = []
    for volume_acc in vol_accs_no_abns:
        idx = np.where(chosen_dataset.volume_accessions == volume_acc)[0][0]
        chosen_dataset_indices.append(idx)
    
    #Init blank dataframe to gather the total
    total_df = pd.DataFrame()
    label_meanings = chosen_dataset.return_label_meanings()
    
    #Now apply the model to each of these scans and get the x_perslice_scores
    for list_position in range(len(chosen_dataset_indices)):
        idx = chosen_dataset_indices[list_position] #int, e.g. 5
        example = chosen_dataset[idx]
        ctvol = example['data'].unsqueeze(0).to(torch.device('cuda:0')) #unsqueeze to create a batch dimension. out shape [1, 135, 3, 420, 420]
        gr_truth = example['gr_truth'].cpu().data.numpy() #out shape [80]
        volume_acc = example['volume_acc'] #this is a string, e.g. 'RHAA12345_5.npz'
        intended_volume_acc = vol_accs_no_abns[list_position]
        assert volume_acc == intended_volume_acc
        print('\tworking on',idx)
        out = model(ctvol)
        x_perslice_scores = out['x_perslice_scores'].cpu().data.numpy()
        this_df = visualize_slicediseases(out['out'], gr_truth, x_perslice_scores, volume_acc, results_dir, label_meanings, None)
        if total_df.shape[0]==0:
            total_df = this_df
        else:
            total_df += this_df
        del example, ctvol, gr_truth, volume_acc, out
    
    #Divide by the total to get the average
    avg_df = total_df/len(chosen_dataset_indices)
    avg_df.to_csv(os.path.join(results_dir,'avg_baseline_df.csv'),header=True,index=True)
    plot_all_organ_heatmaps(avg_df, 'avg_baseline', results_dir)
    return avg_df
    