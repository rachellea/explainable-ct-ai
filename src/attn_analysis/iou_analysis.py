#iou_analysis.py
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
import numpy as np
import pandas as pd
import torch, torch.nn as nn

class DoIOUAnalysis():
    def __init__(self, setname, stop_epoch, label_meanings, results_dir,
                 valid_thresh_perf_df_path):
        """<valid_thresh_perf_df_path> is the path to the validation set file
            created by the method _determine_best_threshold_for_each_label()
            in a previous run. It is only needed if setname=='test' because when
            setname=='test' we need to use the thresholds determined on the
            validation set."""
        self.setname = setname
        assert self.setname in ['valid','test']
        self.stop_epoch = stop_epoch
        self.label_meanings = label_meanings
        self.results_dir = results_dir
        if os.path.exists(valid_thresh_perf_df_path):
            self.valid_thresh_perf_df = pd.read_csv(valid_thresh_perf_df_path, header=0, index_col = 0)
        self.initialize_thresholds_dict()
        self.initialize_iou_wide_df()
        
    ##########################
    # Initialization Methods #--------------------------------------------------
    ##########################
    def initialize_thresholds_dict(self):
        if self.setname == 'valid':
            print('For validation set, using comprehensive base thresholds list')
            self.base_thresholds_list = [0.004,0.008,0.012,0.016,
                                     0.02,0.04,0.06,0.08,0.10,
                                     0.12,0.14,0.16,0.18,0.20,
                                     0.22,0.24,0.26,0.28,0.30,
                                     0.32,0.34,0.36,0.38,0.40,
                                     0.42,0.44,0.46,0.48,0.50,
                                     0.52,0.54,0.56,0.58,0.60,
                                     0.62,0.64,0.66,0.68,0.70,
                                     0.72,0.74,0.76,0.78,0.80,
                                     0.82,0.84,0.86,0.88,0.90,
                                     0.92,0.94,0.96,0.98]
            self.base_thresholds_dict = {}
            for label_name in self.label_meanings:
                self.base_thresholds_dict[label_name] = copy.deepcopy(self.base_thresholds_list)
        elif self.setname == 'test':
            only_best = self.valid_thresh_perf_df[self.valid_thresh_perf_df['Best']=='best']
            #Make sure we only have one threshold per label:
            assert only_best.shape[0]==len(self.label_meanings)
            #example only_best:
            #       Threshold                          Label   MeanIOU    StdIOU  Best
            # 30         0.54                    lung_cancer  0.186747  0.187311  best
            # 89         0.66  heart_coronary_artery_disease  0.437533  0.194666  best
            # 248        0.66       heart_pacemaker_or_defib  0.390899  0.204866  best
            # ...         ...                            ...       ...       ...   ...
            # 4217       0.54                     heart_cabg  0.348482  0.143811  best
            # [80 rows x 5 columns]
            self.base_thresholds_dict = {}
            for index in only_best.index.values.tolist():
                label = only_best.at[index,'Label']
                threshold = only_best.at[index,'Threshold']
                self.base_thresholds_dict[label] = [threshold]
            print('For test set, using only validation set best thresholds',self.base_thresholds_dict)
            
    def initialize_iou_wide_df(self):
         #Path to save df
        self.iou_wide_df_savepath = os.path.join(self.results_dir,'All_IOUs_for_Epoch'+str(self.stop_epoch)+'_MultipleThresholds.csv')
        
        if os.path.exists(self.iou_wide_df_savepath):
            self.loaded_from_existing_file = True
            self.iou_wide_df = pd.read_csv(self.iou_wide_df_savepath,header=0,index_col=False)
            print('Loading iou_wide_df from existing file:',self.iou_wide_df_savepath)
        else:
            self.loaded_from_existing_file = False
            #Initialize iou_wide_df for storing results
            #iou_wide_df includes calculation of the IOU for the
            #various thresholds for each attention map.
            #Make sure it is bigger than it needs to be otherwise the program
            #will choke and take forever to finish.
            totrows = 53*8000*26
            self.iou_wide_df = pd.DataFrame(index=[z for z in range(totrows)],
                                columns=['Epoch','AVal','BVal','IOU','Label','VolumeAccession',
                                         'LabelsPerImage','Threshold'])
            self.curridx = 0
            print('Creating iou_wide_df from scratch')
        
    ####################################################
    # Adding an example to iou_wide_df and final steps #------------------------
    ####################################################
    def add_this_example_to_iou_wide_df(self, segprediction_clipped_and_normed,
                            seg_gr_truth, volume_acc, label_name, num_labels_this_ct):
        """Variables:
        <segprediction_clipped_and_normed> is a np array with shape
            [135, 6, 6] with float values in the range 0.0 - 1.0. It represents
            the attention prediction for this disease.
        <seg_gr_truth> is a np array with shape [135, 6, 6] containing either 0
            or 1. It represents the attention ground truth for this disease.
        <volume_acc> is a string e.g. 'RHAA12345_6.npz'
        <label_name> is a string e.g. 'lung_atelectasis'
        <num_labels_this_ct> is an int representing the total number of labels
            in this CT scan"""
        #Resample the segprediction_clipped_and_normed only if needed
        segprediction_clipped_and_normed = resample_segprediction_only_if_needed(segprediction_clipped_and_normed)
        
        #For the various thresholds in self.base_thresholds_list, calculate the
        #segprediction_bin and the approximate IOU after binarizing:
        label_relevant_thresholds = self.base_thresholds_dict[label_name]
        for threshold in label_relevant_thresholds:
            segprediction_bin = (segprediction_clipped_and_normed>threshold).astype('int')
            
            #Calculate IOU-like quantity
            a_val, b_val = calculate_iou_subcomponents(segprediction_bin, seg_gr_truth)
            if ((a_val!=0) and (b_val!=0)):
                iou = (float(a_val)/float(a_val+b_val))
            else:
                iou = 0
                                
            #Store info in self.iou_wide_df
            self.iou_wide_df.at[self.curridx, 'Epoch'] = self.stop_epoch
            self.iou_wide_df.at[self.curridx, 'AVal'] = a_val
            self.iou_wide_df.at[self.curridx, 'BVal'] = b_val
            self.iou_wide_df.at[self.curridx, 'IOU'] = iou
            self.iou_wide_df.at[self.curridx, 'Label'] = label_name
            self.iou_wide_df.at[self.curridx, 'VolumeAccession'] = volume_acc
            self.iou_wide_df.at[self.curridx, 'LabelsPerImage'] = num_labels_this_ct
            self.iou_wide_df.at[self.curridx, 'Threshold'] = threshold
            self.curridx+=1
            
    def do_all_final_steps(self):
        self._final_clean_of_iou_wide_df()
        if self.setname == 'valid':
            self._determine_best_threshold_for_each_label()
            df_of_best_thresh = self._calculate_df_of_best_thresh_for_valid_set()
        elif self.setname == 'test':
            #The iou_wide_df was already created using only the best threshold
            #for each label as determined by the validation set.
            df_of_best_thresh = self.iou_wide_df
        self._calculate_summary_stats_for_ious(df_of_best_thresh)
    
    ######################
    # Additional Methods #------------------------------------------------------
    ######################
    def _final_clean_of_iou_wide_df(self):
        #Final save - convert to needed dtypes for making figures
        self.iou_wide_df = self.iou_wide_df.dropna(axis=0,how='any')
        self.iou_wide_df['AVal'] = self.iou_wide_df['AVal'].astype('float')
        self.iou_wide_df['BVal'] = self.iou_wide_df['BVal'].astype('float')
        self.iou_wide_df['IOU'] = self.iou_wide_df['IOU'].astype('float')
        self.iou_wide_df['Threshold'] = self.iou_wide_df['Threshold'].astype('float')
        self.iou_wide_df['LabelsPerImage'] = self.iou_wide_df['LabelsPerImage'].astype('int')
        self.iou_wide_df.to_csv(self.iou_wide_df_savepath,header=True,index=False)
    
    def _determine_best_threshold_for_each_label(self):
        """Determine the best threshold for each label based on the mean IOU"""
        #Calculate the mean IOU and std IOU for different labels and thresholds
        thresh_perf_df = pd.DataFrame(index=[x for x in range(3000)],
                                          columns=['Threshold','Label','MeanIOU','StdIOU','Best'])
        curridx = 0
        #Do list(set(self.iou_wide_df['Label'].values.tolist())) instead of
        #self.label_meanings because if you are using a subset of the data
        #then it's possible not all label_meanings are represented.
        for label_name in list(set(self.iou_wide_df['Label'].values.tolist())):
            sel_label = self.iou_wide_df[self.iou_wide_df['Label']==label_name]
            for threshold in self.base_thresholds_list:
                sel_thresh = sel_label[sel_label['Threshold']==threshold]
                #Calculate mean and std of IOU
                thresh_perf_df.at[curridx,'MeanIOU']=np.mean(sel_thresh['IOU'].values)
                thresh_perf_df.at[curridx,'StdIOU']=np.std(sel_thresh['IOU'].values)
                #Fill in basics
                thresh_perf_df.at[curridx,'Threshold']=threshold
                thresh_perf_df.at[curridx,'Label']=label_name
                curridx+=1
        thresh_perf_df = thresh_perf_df.dropna(axis=0,how='all')
        
        #Pick the best threshold per label
        self.best_thresholds = {}
        for label_name in list(set(self.iou_wide_df['Label'].values.tolist())):
            relevant_indices = thresh_perf_df[thresh_perf_df['Label']==label_name].index.values.tolist()
            best_mean_iou = -1*np.inf
            best_mean_iou_index = -1*np.inf
            for idx in relevant_indices:
                idx_mean_iou = thresh_perf_df.at[idx,'MeanIOU']
                if idx_mean_iou > best_mean_iou:
                    best_mean_iou = idx_mean_iou
                    best_mean_iou_index = idx
            self.best_thresholds[label_name]=[thresh_perf_df.at[best_mean_iou_index,'Threshold']]
            thresh_perf_df.at[best_mean_iou_index,'Best'] = 'best'
        thresh_perf_df.to_csv(os.path.join(self.results_dir,'Determine_Best_Threshold_For_Each_Label_Epoch'+str(self.stop_epoch)+'.csv'), header=True, index=True)
        print(thresh_perf_df[thresh_perf_df['Best']=='best'])
    
    def _calculate_df_of_best_thresh_for_valid_set(self):
        """Calculate a dataframe with the same format as the iou_wide_df
        but which contains only the best threshold for each label."""
        assert self.setname == 'valid' #on test set, need to use valid set thresholds
        #For each individual CT volume and label, select the IOU with the
        #best threshold for that label
        #Columns 'Epoch','IOU','Label','VolumeAccession','LabelsPerImage','Threshold'
        df_of_best_thresh = pd.DataFrame(columns=self.iou_wide_df.columns.values.tolist())
        for label_name in list(set(self.iou_wide_df['Label'].values.tolist())):
            sel_label = self.iou_wide_df[self.iou_wide_df['Label']==label_name]
            #in the next line, we do self.best_thresholds[label_name][0] because
            #self.best_thresholds[label_name] is a list with one element e.g. [0.02]
            sel_label_thresh = sel_label[sel_label['Threshold']==self.best_thresholds[label_name][0]]
            df_of_best_thresh = pd.concat([df_of_best_thresh,sel_label_thresh],axis=0,ignore_index=True)
        df_of_best_thresh.to_csv(os.path.join(self.results_dir,'All_IOUs_for_Epoch'+str(self.stop_epoch)+'_BestThresholdOnly.csv'), header=True, index=True)
        return df_of_best_thresh
    
    def _calculate_summary_stats_for_ious(self, df_of_best_thresh):
        self._calculate_summary_stats_by_concept_groups(df_of_best_thresh)
        self._calculate_summary_stats_by_labelsperimage(df_of_best_thresh)
        self._calculate_summary_stats_by_diseaselabel(df_of_best_thresh)
    
    def _calculate_summary_stats_by_concept_groups(self, df_of_best_thresh):
        """Calculate the mean IOU by concept groups.
        concept_groups is a df that has a 'Labels' column
        (e.g. ['lung_atelectasis',...,'heart_cardiomegaly']) and three more
        columns, 'Focal_vs_Diffuse', 'Intervention_vs_Biological', and 'Region'
        which contain tags that divide up the labels into different groups:
        focal vs diffuse, biological vs intervention, and great_vessel vs
        mediastinum vs heart vs lung."""
        #e.g. os.getcwd() is /home/rlb61/data/img-hiermodel2
        concept_groups_path = os.path.join(os.getcwd(),'src/load_dataset/label_proc/conceptual_label_groups.csv')
        concept_groups = pd.read_csv(concept_groups_path,header=0,index_col=False)
        mean_df_by_conceptgroups = pd.DataFrame(index=['focal','diffuse','biological','intervention','great_vessel','mediastinum','heart','lung','overall'],columns=['MeanIOU','StdIOU','Count'])
        for colname in ['Focal_vs_Diffuse', 'Intervention_vs_Biological', 'Region']:
            values = list(set(concept_groups[colname].values.tolist()))
            for value in values: #e.g. value='focal' or 'diffuse' for colname 'Focal_vs_Diffuse'
                diseases_with_that_value = concept_groups[concept_groups[colname]==value]['Label']
                sel_ious = df_of_best_thresh[df_of_best_thresh['Label'].isin(diseases_with_that_value)]['IOU']
                mean_df_by_conceptgroups.at[value,'MeanIOU'] =  np.mean(sel_ious.values)
                mean_df_by_conceptgroups.at[value,'StdIOU'] = np.std(sel_ious.values)
                mean_df_by_conceptgroups.at[value,'Count'] = sel_ious.values.size
        #Calculate the mean IOU overall
        mean_iou_overall = np.mean(df_of_best_thresh['IOU'].values)
        std_iou_overall = np.std(df_of_best_thresh['IOU'].values)
        print('For Epoch',self.stop_epoch,'the mean IOU overall, after the best threshold per label has been selected, is',
              round(mean_iou_overall,3),'+/-',round(std_iou_overall,3))
        mean_df_by_conceptgroups.at['overall','MeanIOU'] =  mean_iou_overall
        mean_df_by_conceptgroups.at['overall','StdIOU'] = std_iou_overall
        mean_df_by_conceptgroups.at['overall','Count'] = df_of_best_thresh['IOU'].values.size
        mean_df_by_conceptgroups.to_csv(os.path.join(self.results_dir,'Summary_of_IOUs_for_Epoch'+str(self.stop_epoch)+'_BestThresholdOnly_ByConceptGroups.csv'),header=True,index=True)
    
    def _calculate_summary_stats_by_labelsperimage(self, df_of_best_thresh):
        #Calculate the mean IOU for each LabelsPerImage
        max_labelsperimage = np.max(self.iou_wide_df['LabelsPerImage'].values)
        mean_df_by_labelsperimage = pd.DataFrame(index=['LabelsPerImage'+str(x) for x in range(1,max_labelsperimage+1)],columns=['MeanIOU','StdIOU','Count'])
        for labels_per_image in range(1,max_labelsperimage+1):
            sel_ious = df_of_best_thresh[df_of_best_thresh['LabelsPerImage']==labels_per_image]['IOU']
            mean_df_by_labelsperimage.at['LabelsPerImage'+str(labels_per_image),'MeanIOU'] = np.mean(sel_ious.values)
            mean_df_by_labelsperimage.at['LabelsPerImage'+str(labels_per_image),'StdIOU'] = np.std(sel_ious.values)
            mean_df_by_labelsperimage.at['LabelsPerImage'+str(labels_per_image),'Count'] = sel_ious.values.size
        mean_df_by_labelsperimage.to_csv(os.path.join(self.results_dir,'Summary_of_IOUs_for_Epoch'+str(self.stop_epoch)+'_BestThresholdOnly_ByLabelsPerImage.csv'),header=True,index=True)
    
    def _calculate_summary_stats_by_diseaselabel(self, df_of_best_thresh):
        #Calculate the mean IOU for each Label (each 'disease')
        mean_df_by_disease = pd.DataFrame(index=list(set(self.iou_wide_df['Label'].values.tolist())),columns=['MeanIOU','StdIOU','Count'])
        for label in list(set(self.iou_wide_df['Label'].values.tolist())):
            sel_ious = df_of_best_thresh[df_of_best_thresh['Label']==label]['IOU']
            mean_df_by_disease.at[label,'MeanIOU'] = np.mean(sel_ious.values)
            mean_df_by_disease.at[label,'StdIOU'] = np.std(sel_ious.values)
            mean_df_by_disease.at[label,'Count'] = sel_ious.values.size
        mean_df_by_disease.to_csv(os.path.join(self.results_dir,'Summary_of_IOUs_for_Epoch'+str(self.stop_epoch)+'_BestThresholdOnly_ByDiseaseLabel.csv'),header=True,index=True)


def calculate_iou_subcomponents(segprediction, seg_gr_truth):
    """Calculate components that can be used to calculate an
    intersection-over-union-like quantity. Returns:
        a_val: the sum of the segprediction values in the allowed area
            (i.e. in the area where seg_gr_truth==1)
        b_val: the sum of the segprediction values in the forbidden area
            (i.e. in the area where seg_gr_truth==0)
    
    Variables:
    <segprediction>: a 2D numpy array of 1s and 0s indicating the predicted
        segmentation mask, for a particular label. The values are 1 where the
        pixels are predicted to be part of the label, and 0 otherwise. Example:
            0 0 0 1 1
            0 0 1 1 1
            0 0 1 1 0
            0 0 1 0 0
    <seg_gr_truth>: a 2D numpy array of integers indicating the ground truth
        allowed region, which is based on organ. Therefore, the segprediction
        should be within the allowed region (within the organ, where the
        seg_gr_truth is 1). Example:
            0 1 1 1 1
            0 0 1 1 1
            0 0 1 1 1
            0 0 1 1 0"""
    #Sum of predictions in allowed region:
    a_val = float(np.sum(segprediction[seg_gr_truth==1]))
    #Sum of predictions in forbidden region:
    b_val = float(np.sum(segprediction[seg_gr_truth==0]))
    return a_val, b_val

def resample_segprediction_only_if_needed(segpred):
    """For the CTNetModel model, the segpred has shape (135, 14, 14)
    which must be resampled to (135, 6, 6) in order to match the seg_gr_truth.
    In an ideal world all approximate IOU computations would take place in the
    space of the original CT scan but that is prohibitively computationally
    expensive (~1 month of runtime for a single model's IOU results)"""
    if segpred.shape == (135,6,6): #then do nothing
        return segpred
    #Other shapes:
    #(135,14,14) is from CTNetModel
    #(5,5,5) is from BodyConv
    elif segpred.shape in [(135,14,14), (5,5,5)]: #resample to (135,6,6).
        new_segpred = nn.functional.interpolate(torch.Tensor(segpred).unsqueeze(0).unsqueeze(0), size=[135,6,6], mode='trilinear').squeeze()
        return new_segpred.numpy()
    else:
        assert False, 'Invalid shape for segprediction_clipped_and_normed'
    