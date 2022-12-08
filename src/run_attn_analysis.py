#run_attention_analysis.py
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
import torch
import datetime
import numpy as np
import pandas as pd

from src.attn_analysis import gradcam
from src.attn_analysis import iou_analysis
from src.attn_analysis import blue_heatmap
from src.attn_analysis import extract_disease_reps
from src.attn_analysis import make_2d_plot_and_3d_gif

import warnings
warnings.filterwarnings('ignore')

class AttentionAnalysis(object):
    def __init__(self, results_dir_force,
                 base_results_dir, task,
                 attention_type, attention_type_args,
                 setname, valid_results_dir,
                 custom_net, custom_net_args, params_path,
                 stop_epoch, which_scans, dataset_class, dataset_args):
        """
        Variables:
        <results_dir_force>: path to a results directory. If this is a valid
            path, then all results will be stored in here. If this is NOT a
            valid path, a new directory for the new results will be created
            based on <base_results_dir>.
        <base_results_dir>: path to the base results directory. A new directory
            will be created within this directory to store the results of
            this experiment.
        <task>: a list of strings. The strings may include 'iou_analysis',
            'blue_heatmaps', and/or 'attn_plots'.
            
            If <task> contains 'iou_analysis' then calculate approximate IOU
            statistics for the final epoch of a model.
            Specifically, the 'IOU' is calculated as the ratio of raw scores
            within the allowed area to raw scores outside of the allowed
            area.
            Produces iou_wide_df, a dataframe with the following 5 columns:
            'Epoch': int, the epoch in which the IOU was calculated.
            'IOU': float, the 'IOU' value for this label's attention map vs. the
                segmentation ground truth (which in this case is the approximate
                attention ground truth.)
            'Label': string for the label for which IOU was calculated e.g. 'airplane'
            'VolumeAccession': volume accession number
            'LabelsPerImage': total number of labels present in this image
            Also produces dfs that summarize the IOU across different ways
            of grouping the data.
            
            If <task> contains 'blue_heatmaps' then make a blue heatmap showing
            the disease scores for each slice.
            
            If <task> contains 'attn_plots' then make visualizations of the
            attention superimposed on the CT scan (as a 3D gif, and as a 2D plot
            for the slice with the highest score for that disease). Also if
            doing Grad-CAM, make a 2d debugging plot.
            
        <attention_type>: str; either
            'gradcam-vanilla' for vanilla Grad-CAM, or
            'hirescam' for HiResCAM, in which feature maps and gradients are
                element-wise multiplied and then we take the avg over the
                feature dimension, or
            'hirescam-check' for alternative implementation of HiResCAM
                attention calculation, which can be used in a model that
                has convolutional layers followed by a single FC layer.
                In this implementation, the HiResCAM attention is calculated
                during the forward pass of the model by element-wise multiplying
                the final FC layer weights (the gradients) against the final
                representation. This option is called 'hirescam-check'
                because for models that meet the architecture requirements this
                implementation is a 'check' on the 'hirescam' option which
                actually accesses the gradients.
                'hirescam-check' and 'hirescam' on the output of the last conv
                layer produce identical results on AxialNet as expected, since
                AxialNet is a CNN with one FC layer at the end.
        <attention_type_args>: dict; additional arguments needed to calculate
            the specified kind of attention. If the attention_type is one of the
            GradCAMs then in this dict we need to specify 
            'model_name' and 'target_layer_name' (see gradcam.py for
            more documentation)
        <setname>: str; which split to use e.g. 'train' or 'val' or 'test'; will
            be passed to the <dataset_class>
        <valid_results_dir>: path to a directory that contains the validation
            set IOU analysis results. Only needed if setname=='test' because we
            need to use validation set per-label thresholds to calculate
            results. 
        <custom_net>: a PyTorch model
        <custom_net_args>: dict; arguments to pass to the PyTorch model
        <params_path>: str; path to the model parameters that will be loaded in
        <stop_epoch>: int; epoch at which the model saved at <params_path> was
            saved
        <which_scans>: a pandas DataFrame specifying what scans and/or
            abnormalities to use.
            It can be an empty pandas DataFrame, in which case all available
            scans in the set will be used and named with whatever volume
            accession they were saved with (real or fake).
            Or, it can be a filled in pandas DataFrame, with columns
            ['VolumeAcc','VolumeAcc_ForOutput','Abnormality'] where
            VolumeAcc is the volume accession the scan was saved with,
            VolumeAcc_ForOutput is the volume accession that should be used in
            the file name of any output files of this module (e.g. a DEID acc),
            and Abnormality is either 'all' to save all abnormalities for that
            scan, or it's comma-separated names of specific abnormalities to
            save for that scan.
        <dataset_class>: a PyTorch dataset class
        <dataset_args>: dict; arguments to pass to the <dataset_class>"""
        self.base_results_dir = base_results_dir
        self.task = task
        for specific_task in self.task:
            assert ((specific_task == 'iou_analysis')
                or (specific_task == 'blue_heatmaps')
                or (specific_task == 'attn_plots'))
        assert len(self.task) <= 2
        if 'blue_heatmaps' in self.task:
            #only allow calculation of the blue_heatmaps if we are using
            #attention_type hirescam-check. Why? Because for both the blue
            #heatmaps and the hirescam-check visualizations, we need to run
            #the model to get out. And in gradcam we need to run the model again
            #later so we get a memory error if we try to do this after getting
            #out.
            assert attention_type == 'hirescam-check'
        self.attention_type = attention_type
        assert self.attention_type in ['gradcam-vanilla','hirescam','hirescam-check']
        self.attention_type_args = attention_type_args
        if self.attention_type in ['gradcam-vanilla','hirescam']:
            assert 'model_name' in self.attention_type_args.keys()
            assert 'target_layer_name' in self.attention_type_args.keys()
        self.setname = setname
        self.valid_results_dir = valid_results_dir
        self.custom_net = custom_net
        self.custom_net_args = custom_net_args #dict of args
        self.params_path = params_path
        self.stop_epoch = stop_epoch
        self.which_scans = which_scans
        self.CTDatasetClass = dataset_class
        self.dataset_args = dataset_args #dict of args
        self.device = torch.device('cuda:0')
        self.verbose = self.dataset_args['verbose'] #True or False
        
        #Run
        self.set_up_results_dirs(results_dir_force)
        self.run()
    
    def set_up_results_dirs(self, results_dir_force):
        if os.path.isdir(results_dir_force):
            results_dir = results_dir_force
        else:
            #If you're not forcing a particular results_dir, then make a new
            #results dir:
            #Example params_path = '/home/rlb61/data/img-hiermodel2/results/2020-09/2020-09-27_AxialNet_Mask_CORRECT_dilateFalse_nearest/params/AxialNet_Mask_CORRECT_dilateFalse_nearest_epoch23'
            old_results_dir = os.path.split(os.path.split(os.path.split(self.params_path)[0])[0])[1] #e.g. '2020-09-27_AxialNet_Mask_CORRECT_dilateFalse_nearest'
            date = datetime.datetime.today().strftime('%Y-%m-%d')
            results_dir = os.path.join(self.base_results_dir,date+'_'+self.setname.capitalize()+'AttnAnalysis_of_'+old_results_dir)
            if not os.path.isdir(results_dir):
                os.mkdir(results_dir)
        
        #Subdirs for particular analyses:
        if 'iou_analysis' in self.task:
            self.iou_analysis_dir = os.path.join(results_dir,'iou_analysis_'+self.attention_type)
            if not os.path.exists(self.iou_analysis_dir): os.mkdir(self.iou_analysis_dir)
        if 'blue_heatmaps' in self.task:
            #Note that the blue heatmaps depend only on the model, and not on the
            #attention type
            self.blue_heatmaps_dir = os.path.join(results_dir,'blue_heatmaps')
            if not os.path.exists(self.blue_heatmaps_dir): os.mkdir(self.blue_heatmaps_dir)
        if 'attn_plots' in self.task:
            self.attn_2dplot_dir = os.path.join(results_dir,'attn_2dplot_'+self.attention_type)
            self.attn_3dgif_dir = os.path.join(results_dir,'attn_3dgif_dir_'+self.attention_type)
            for directory in [self.attn_2dplot_dir,self.attn_3dgif_dir]:
                if not os.path.exists(directory): os.mkdir(directory)
            for key in ['g1p1', 'g1p0', 'g0p1', 'g0p0']:
                if not os.path.exists(os.path.join(self.attn_2dplot_dir,key)):
                    os.mkdir(os.path.join(self.attn_2dplot_dir,key))
                if not os.path.exists(os.path.join(self.attn_3dgif_dir,key)):
                    os.mkdir(os.path.join(self.attn_3dgif_dir,key))
            
            if self.attention_type in ['gradcam-vanilla','hirescam']:
                self.gradcam_debug_dir = os.path.join(results_dir,self.attention_type+'_debug_dir')
                if not os.path.exists(self.gradcam_debug_dir): os.mkdir(self.gradcam_debug_dir)
        else: #even if attn_plots is not in task, we need to have a placeholder for
            #this directory to avoid an error later:
            self.gradcam_debug_dir = None
    
    def run(self):
        self.load_model()
        self.load_dataset()
        self.load_chosen_indices()
        if 'blue_heatmaps' in self.task:
            self.blue_heatmap_baseline = blue_heatmap.get_baseline(self.chosen_dataset, self.model, self.blue_heatmaps_dir)
        if 'iou_analysis' in self.task:
            thresh_perf_df_filename = 'Determine_Best_Threshold_For_Each_Label_Epoch'+str(self.stop_epoch)+'.csv'
            valid_thresh_perf_df_path = os.path.join(os.path.join(self.valid_results_dir,'iou_analysis_'+self.attention_type), thresh_perf_df_filename)
            self.iou_analysis_object = iou_analysis.DoIOUAnalysis(self.setname, self.stop_epoch,
                                    self.label_meanings, self.iou_analysis_dir, valid_thresh_perf_df_path)
        self.loop_over_dataset_and_labels()
        if 'iou_analysis' in self.task:
             self.iou_analysis_object.do_all_final_steps()
    
    ######################################################
    # Methods to Load Model, Dataset, and Chosen Indices #----------------------
    ######################################################
    def load_model(self):
        print('Loading model')
        self.model = self.custom_net(**self.custom_net_args).to(self.device)
        check_point = torch.load(self.params_path, map_location='cpu') #map to CPU to avoid memory issue #TODO check if you need this
        self.model.load_state_dict(check_point['params'])
        self.model.eval()
        #If everything loads correctly you will see the following message:
        #IncompatibleKeys(missing_keys=[], unexpected_keys=[])
    
    def load_dataset(self):
        print('Loading dataset')
        self.chosen_dataset = self.CTDatasetClass(setname = self.setname, **self.dataset_args)
        self.label_meanings = self.chosen_dataset.return_label_meanings()
    
    def load_chosen_indices(self):
        print('Loading chosen indices')
        if len([x for x in self.which_scans.columns.values.tolist() if x in ['VolumeAcc','VolumeAcc_ForOutput','Abnormality']])==3:        
            #you did specify which scans to use, so figure out what indices
            #you need to query in the dataset to get those chosen scans:
            for df_idx in range(self.which_scans.shape[0]):
                volume_acc = self.which_scans.at[df_idx,'VolumeAcc']
                self.which_scans.at[df_idx,'ChosenIndex'] = np.where(self.chosen_dataset.volume_accessions == volume_acc)[0][0]
        else:
            assert (self.which_scans == pd.DataFrame()).all().all()
            #you didn't specify which scans to use, so use all the scans in the dataset
            self.which_scans['ChosenIndex'] = [x for x in range(len(self.chosen_dataset))]
        self.which_scans['ChosenIndex'] = self.which_scans['ChosenIndex'].astype('int')
    
    ###########
    # Looping #-----------------------------------------------------------------
    ###########
    def loop_over_dataset_and_labels(self):
        if (self.task == ['iou_analysis'] and self.iou_analysis_object.loaded_from_existing_file):
            return #don't need to loop again if iou_wide_df already created
        print('Looping over dataset and labels')
        five_percent = max(1,int(0.05*self.which_scans.shape[0]))
        #Iterate through the examples in the dataset. df_idx is an integer
        for df_idx in range(self.which_scans.shape[0]):
            if self.verbose: print('Starting df_idx',df_idx)
            idx = self.which_scans.at[df_idx,'ChosenIndex'] #int, e.g. 5
            example = self.chosen_dataset[idx]
            ctvol = example['data'].unsqueeze(0).to(self.device) #unsqueeze to create a batch dimension. out shape [1, 135, 3, 420, 420]
            gr_truth = example['gr_truth'].cpu().data.numpy() #out shape [80]
            volume_acc = example['volume_acc'] #this is a string, e.g. 'RHAA12345_5.npz'
            attn_gr_truth = example['attn_gr_truth'].data.cpu().numpy() #out shape [80, 135, 6, 6]
            
            #Get out and x_perslice_scores when using attention_type hirescam-check
            out = self.get_out_and_blue_heatmaps(ctvol, gr_truth, volume_acc)
            
            if self.verbose: print('Analyzing',volume_acc)
            #volume_acc sanity check and conversion to FAKE volume acc if indicated
            if 'VolumeAcc' in self.which_scans.columns.values.tolist():
                intended_volume_acc = self.which_scans.at[df_idx,'VolumeAcc']
                assert volume_acc == intended_volume_acc
                #Now, because which_scans is not empty, you can switch volume_acc
                #from the actual volume acc e.g. RHAA12345_6 to the fake ID,
                #because from here onwards, the volume acc is only used in file
                #names:
                volume_acc = self.which_scans.at[df_idx,'VolumeAcc_ForOutput'].replace('.npz','').replace('.npy','') #e.g. fake ID 'val12345'
            
            #Now organize the labels for this particular image that you want to
            #make heatmap visualizations for into g1p1, g1p0, g0p1, and g0p0
            #g1p1=true positive, g1p0=false negative, g0p1=false positive, g0p0=true negative
            #we pass in volume_acc twice because the variable volume_acc could
            #be fake OR real, depending on the preceding logic, but
            #example['volume_acc'] is guaranteed to always be real.
            label_indices_dict = make_label_indices_dict(volume_acc, example['volume_acc'], gr_truth, self.params_path, self.label_meanings)
            
            for key in ['g1p1', 'g1p0', 'g0p1', 'g0p0']:
                chosen_label_indices = label_indices_dict[key] #e.g. [32, 37, 43, 46, 49, 56, 60, 62, 64, 67, 68, 71]
                
                if (('Abnormality' not in self.which_scans.columns.values.tolist()) or (self.which_scans.at[df_idx,'Abnormality'] == 'all')): #plot ALL abnormalities
                    pass 
                else: #plot only chosen abnormalities
                    chosen_abnormalities = self.which_scans.at[df_idx,'Abnormality'].split(',')
                    chosen_label_indices = [x for x in chosen_label_indices if self.label_meanings[x] in chosen_abnormalities]
                
                #Calculate label-specific attn and make label-specific attn figs
                for chosen_label_index in chosen_label_indices:
                    #Get label_name and seg_gr_truth:
                    label_name = self.label_meanings[chosen_label_index] #e.g. 'lung_atelectasis'
                    seg_gr_truth = attn_gr_truth[chosen_label_index,:,:,:] #out shape [135, 6, 6]
                    #segprediction is the raw attention. slice_idx is the index of
                    #the slice with the highest raw score for this label
                    segprediction, x_perslice_scores_this_disease = self.return_segprediction(out, ctvol, gr_truth, volume_acc, chosen_label_index) #out shape [135, 6, 6]
                    segprediction_clipped_and_normed = clip_and_norm_volume(segprediction)
                    
                    if 'iou_analysis' in self.task:
                        if key in ['g1p1','g1p0']: #TODO: implement IOU analysis for other options! also make this more efficient so no excessive calculations are done
                            if self.verbose: print('Adding example to IOU analysis')
                            self.iou_analysis_object.add_this_example_to_iou_wide_df(segprediction_clipped_and_normed,
                                                    seg_gr_truth, volume_acc, label_name, num_labels_this_ct=int(gr_truth.sum()))
                    if 'attn_plots' in self.task:
                        if self.verbose: print('Making 2D and 3D attn figures')
                        make_2d_plot_and_3d_gif.plot_attn_over_ct_scan(ctvol,
                            segprediction_clipped_and_normed, x_perslice_scores_this_disease, volume_acc,
                            label_name, os.path.join(self.attn_2dplot_dir,key), os.path.join(self.attn_3dgif_dir,key))
            
            #Report progress
            if df_idx % five_percent == 0:
                print('Done with',df_idx,'=',round(100*df_idx/self.which_scans.shape[0],2),'%')
            del example, ctvol, gr_truth, volume_acc, attn_gr_truth, out
    
    def get_out_and_blue_heatmaps(self, ctvol, gr_truth, volume_acc):
        """Calculate 'out' which will be used for:
            1. the blue heatmap figure (the 'x_perslice_scores') which is
               specific to a particular scan, NOT a particular label;
            2. the 'hirescam-check' attention (the 'disease_reps')
        Note that we don't do this within the label for loop below
        because it's computationally wasteful to run a fixed model again
        and again on the same input CT scan.
        
        To avoid memory issues of running the model twice,
        for determining true positives/false positives/true negatives/false
        negatives, we use the pre-calculated predicted probabilities that were
        saved when the model was first run.
        
        out['out'] contains the prediction scores and has shape [1,80]
        out['disease_reps'] contains the 'hirescam-check' attention for
            all diseases and has shape [80, 135, 16, 6, 6]
        out['x_perslice_scores'] contains the abnormality scores for each
            slice and has shape [1, 80, 135]"""
        if self.attention_type == 'hirescam-check':
            out = self.model(ctvol)
            if 'blue_heatmaps' in self.task:
                if self.verbose: print('Making blue heatmap')
                blue_heatmap.visualize_slicediseases(out['out'], gr_truth,
                                    out['x_perslice_scores'].cpu().data.numpy(),
                                    volume_acc, self.blue_heatmaps_dir, self.label_meanings,
                                    self.blue_heatmap_baseline)
            return out
        else:
            return None
    
    def return_segprediction(self, out, ctvol, gr_truth, volume_acc, chosen_label_index):
        """Return the <segprediction> which is a volume of scores for a particular
        label"""
        if self.attention_type == 'hirescam-check':
            return extract_disease_reps.return_segprediction_from_disease_rep(out, chosen_label_index)
        elif self.attention_type in ['gradcam-vanilla','hirescam']:
            #note that if 'make_figure' is in self.task, then a 2d debugging
            #figure for Grad-CAM will also be saved in this step
            return gradcam.RunGradCAM(self.attention_type, self.model, self.device,
                      self.label_meanings, self.gradcam_debug_dir, self.task,
                      **self.attention_type_args).return_segprediction_from_grad_cam(ctvol, gr_truth, volume_acc, chosen_label_index)

def make_label_indices_dict(possibly_fake_volume_acc, real_volume_acc, gr_truth, params_path, label_meanings):
    """Based on the <gr_truth> and the predicted probability that was
    pre-calculated, figure out which abnormalities are true positives (g1p1),
    false negatives (g1p0), false positives (g0p1), and true negatives (g0p0).
    g stands for ground truth and p stands for predicted probability.
    
    The predicted probabilities are read in from the predicted probabilities
    that were saved from the final model when it was done training.
    The path for these is inferred from params_path based on known
    directory structure. We also need to use this pre-calculated file because
    we need to get the median predicted probability for each abnormality.
    
    The predicted probabilities are binarized as 0 or 1 according to being
    above or below the median (50th percentile) for that abnormality.
    
    Returns a dictionary with keys g1p1, g1p0, g0p1, and g0p0
    and values that are numpy arrays of numeric indices of the corresponding
    abnormalities e.g. array([32, 37, 64, 67, 68, 71])"""
    #Infer paths to the precomputed pred probs based on known directory organization:
    #e.g. precomputed_path = '/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart/pred_probs'
    precomputed_path = os.path.join(os.path.split(os.path.split(params_path)[0])[0],'pred_probs')
    files = os.listdir(precomputed_path) #e.g. ['valid_grtruth_ep4.csv', 'valid_predprob_ep4.csv']
    pred_probs_file = [x for x in files if 'predprob' in x][0] #e.g. 'valid_predprob_ep4.csv'
    gr_truth_file = [x for x in files if 'grtruth' in x][0] #e.g. 'valid_grtruth_ep4.csv'
    
    #Open the pred probs and gr truth for this data subset
    #Each of them has volume accesions as the index, and abnormalities as
    #the columns. Example shape: [2085,80]
    pred_probs_all = pd.read_csv(os.path.join(precomputed_path, pred_probs_file),header=0,index_col=0)
    gr_truth_all = pd.read_csv(os.path.join(precomputed_path, gr_truth_file),header=0,index_col=0)
    
    #Sanity checks:
    for df in [pred_probs_all, gr_truth_all]:
        assert df.columns.values.tolist()==label_meanings
    assert (gr_truth_all.loc[real_volume_acc,:]==gr_truth).all()
    
    #Calculate the medians of the different abnormalities across the whole
    #data subset. 
    medians = np.median(pred_probs_all,axis=0) #np array, e.g. shape [80]
    
    #Select out the predicted probabilities for just this scan
    pred_probs = pred_probs_all.loc[real_volume_acc,:] #pd Series w abn labels and float values, e.g. shape [80]
    
    #Get binary vector that's equal to 1 if the corresponding abnormality
    #has a pred prob greater than the median
    pred_probs_geq = (pred_probs >= medians).astype('int') #pd Series w abn labels and binary int values, e.g. shape [80]
    
    #Now divide up the abnormalities for this particular CT scan based on whether
    #they are above or below the median pred prob, and whether the gr truth
    #is 1 or 0
    g0p0 = np.intersect1d(np.where(gr_truth==0)[0], np.where(pred_probs_geq==0)[0])
    g0p1 = np.intersect1d(np.where(gr_truth==0)[0], np.where(pred_probs_geq==1)[0])
    g1p0 = np.intersect1d(np.where(gr_truth==1)[0], np.where(pred_probs_geq==0)[0])
    g1p1 = np.intersect1d(np.where(gr_truth==1)[0], np.where(pred_probs_geq==1)[0])
    
    #Checks
    assert len(g1p0)+len(g1p1)==int(gr_truth.sum())
    assert len(g0p0)+len(g0p1)+len(g1p0)+len(g1p1)==len(gr_truth)
    
    label_indices_dict = {'g0p0':g0p0.tolist(),
                          'g0p1':g0p1.tolist(),
                          'g1p0':g1p0.tolist(),
                          'g1p1':g1p1.tolist()}
    #uncomment the next line to print detailed info to the terminal:
    #print_for_future_reference(params_path, label_indices_dict, possibly_fake_volume_acc, pred_probs, medians, label_meanings)
    return label_indices_dict

def print_for_future_reference(params_path, label_indices_dict, possibly_fake_volume_acc, pred_probs, medians, label_meanings):
    model_description = os.path.split(params_path)[1]
    for key in list(label_indices_dict.keys()): #the keys are ['g0p0','g0p1','g1p0','g1p1']
        for idx in label_indices_dict[key]:
            print('\t'.join([model_description, possibly_fake_volume_acc, key, label_meanings[idx], str(round(pred_probs[idx],4)),'median:',str(round(medians[idx],4))]))

#############
# Functions #-------------------------------------------------------------------
#############
def clip_and_norm_volume(volume):
    volume = np.maximum(volume, 0) #ReLU operation
    volume = volume - np.min(volume)
    if np.max(volume)!=0:
        volume = volume / np.max(volume)
    return volume
