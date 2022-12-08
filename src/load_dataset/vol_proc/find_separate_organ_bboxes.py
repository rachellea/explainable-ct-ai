#find_separate_organ_bboxes.py
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

"""This module is used to calculate bounding box coordinates for the right
lung, left lung, and heart separately. Note that this is DIFFERENT from the
bounding box coordinates saved in extrema.csv by the code in seg/segmentlungs.py.
The coordinates in extrema.csv are for a bounding box that surrounds all
three organs together.
The output file produced by this module, organ_bboxes.csv, is used in
load_dataset/mask.py which in turn is used for mask loss models."""

import timeit
import numpy as np
import pandas as pd

from skimage.measure import label,regionprops

from src.load_dataset.vol_proc import mask
from src.load_dataset.vol_proc import ctvol_preproc

class GetOrganBBoxes(object):
    def __init__(self, extrema_path='/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/extrema_DEID.csv',
                 segmap_path='/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-segmaps-DEID/',
                 metadata_path='/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/CT_Scan_Metadata_Small_DEID.csv'):
        """Calculate right lung, heart, and left lung bounding boxes and save
        their coordinates in an output file organ_bboxes.csv.
        organ_bboxes.csv will then be used in mask.py to determine when part
        of a lung is missing, so that a heuristic segmentation can be used
        for that lung instead of the incorrect segmentation.
        Note that this calculation can only be run after all the lungs have
        been segmented, i.e. after seg/segmentlungs.py has been run to save
        lung segmentation masks of all the CT scans.
        
        Variables:
        <extrema_path> is the path to the extrema CSV file which has the
            following columns: ['VolumeAcc_DEID', 'NoteAcc_DEID', 'MRN_DEID',
            'Set_Assigned','Subset_Assigned','sup_axis0min','inf_axis0max',
            'ant_axis1min','pos_axis1max','rig_axis2min','lef_axis2max',
            'shape0','shape1','shape2']
        <segmap_path>: this is the path to the directory that
            contains all of the precomputed binary lung segmentation maps
            saved as numpy arrays.
        <metadata_path>: path to the CT_Scan_Metadata_Small_DEID.csv file"""
        self.extrema = ctvol_preproc.load_extrema(extrema_path)
        self.segmap_path = segmap_path
        metadata = pd.read_csv(metadata_path,header=0,index_col='VolumeAcc_DEID')
        metadata = metadata[metadata['status']=='success']
        volume_accs = metadata.index.values.tolist()
        
        #Set up self.organ_bboxes which is a dataframe that will store the
        #organ-specific bounding box coordinates. The index is volume_accs and
        #the organ-specific bounding box coordinates are the columns.
        organs = ['RL_','LL_','heart_']
        borders = ['sup_axis0min','inf_axis0max','ant_axis1min','pos_axis1max',
                   'rig_axis2min','lef_axis2max']
        eqt = ['RLsupEqT','RLinfEqT','RLantEqT','RLposEqT','RLrigEqT','RLlefEqT']
        column_names = ([organname+bordername for organname in organs for bordername in borders]+eqt)
        self.organ_bboxes = pd.DataFrame(index = volume_accs, columns = column_names)
        for colname in eqt:
            self.organ_bboxes[colname]=''
        
        #Fill in self.organ_bboxes
        self.fill_in_organ_bbox_data()
    
    def fill_in_organ_bbox_data(self):
        tot0 = timeit.default_timer()
        bad_quality_mask_count = 0
        ok_quality_mask_count = 0
        gr_50px_diff_Rinf_Linf = 0
        gr_100px_diff_Rinf_Linf = 0
        for idx in range(self.organ_bboxes.shape[0]):
            volume_acc = self.organ_bboxes.index.values.tolist()[idx]
            full_size_mask = mask.load_raw_mask_and_split_it_into_organs(volume_acc, self.segmap_path, self.extrema)
            organ_segs = {'RL':(full_size_mask==1), #right_lung_seg. out shape of binary mask [1,bigslices,bigsquare,bigsquare] e.g. [1,45,420,420]
                          'LL':(full_size_mask==3), #left_lung_seg
                          'heart':(full_size_mask==2)} #heart_seg
            for organ in organ_segs.keys():
                if np.sum(full_size_mask)==0: #bad quality mask
                    lef=-1; rig=-1; ant=-1; pos=-1; inf=-1; sup=-1
                else:
                    lef, rig, ant, pos, inf, sup =  find_binary_blob_extrema(organ_segs[organ])
                self.organ_bboxes.at[volume_acc,organ+'_sup_axis0min']=sup
                self.organ_bboxes.at[volume_acc,organ+'_inf_axis0max']=inf
                self.organ_bboxes.at[volume_acc,organ+'_ant_axis1min']=ant
                self.organ_bboxes.at[volume_acc,organ+'_pos_axis1max']=pos
                self.organ_bboxes.at[volume_acc,organ+'_rig_axis2min']=rig
                self.organ_bboxes.at[volume_acc,organ+'_lef_axis2max']=lef
            
            #Checks across organs
            checks = [['sup',min,'0min'], #side, func, axis
                      ['ant',min,'1min'],
                      ['rig',min,'2min'],
                      ['inf',max,'0max'],
                      ['pos',max,'1max'],
                      ['lef',max,'2max']]
            if np.sum(full_size_mask)!=0:
                ok_quality_mask_count+=1
                for side, func, axis in checks:
                    if func(self.organ_bboxes.at[volume_acc,'RL_'+side+'_axis'+axis],self.organ_bboxes.at[volume_acc,'LL_'+side+'_axis'+axis])==self.extrema.at[volume_acc,side+'_axis'+axis]:
                        self.organ_bboxes.at[volume_acc,'RL'+side+'EqT'] = 'Yes' #should be yes basically all the time, unless something is infinite
                    else:
                        self.organ_bboxes.at[volume_acc,'RLs'+side+'EqT'] = 'No'
            else:
                bad_quality_mask_count+=1
            
            #Reporting potentially bad masks based on RL LL difference
            RLdiff = abs(self.organ_bboxes.at[volume_acc,'RL_inf_axis0max'] - self.organ_bboxes.at[volume_acc,'LL_inf_axis0max'])
            if RLdiff > 50:
                gr_50px_diff_Rinf_Linf+=1
            if  RLdiff > 100:
                gr_100px_diff_Rinf_Linf+=1
            
            if ((idx>0) and (idx % 50 == 0)):
                print('Done with',idx,'=',round(100*idx/self.organ_bboxes.shape[0],2),'percent')
                tot1 = timeit.default_timer()
                print('\tElapsed Time', round((tot1 - tot0)/60.0,2),'minutes')
                print('\tbad quality:',bad_quality_mask_count,'=',round(100*bad_quality_mask_count/idx,2),'percent')
                print('\tok quality:',ok_quality_mask_count,'=',round(100*ok_quality_mask_count/idx,2),'percent')
                print('\t\tgreater than 50 px difference Rinf Linf:',gr_50px_diff_Rinf_Linf,'=',round(100*gr_50px_diff_Rinf_Linf/idx,2),'percent')
                print('\t\tgreater than 100 px difference Rinf Linf:',gr_100px_diff_Rinf_Linf,'=',round(100*gr_100px_diff_Rinf_Linf/idx,2),'percent')
                self.organ_bboxes.to_csv('organ_bboxes.csv',header=True,index=True)    
        self.organ_bboxes.to_csv('organ_bboxes.csv',header=True,index=True)
        print('Done')

def find_binary_blob_extrema(segmap):
    """Return the superior, inferior, anterior, posterior, left, and
    right bounding box coordinates for the bbox surrounding the blob of ones
    inside the binary np array <segmap>.
    Use case: the blob of ones may represent the right lung, or the heart,
    or the left lung. The point of this function is to enable
    figuring out whether the right lung or left lung has its bottom cut off.
    The blob of ones should have the default axial orientation of a CT volume.
    This function shares the same key logic as find_lung_extrema() in
    segmentlungs.py"""
    #Bounding box coordinates:
    label_image = label(segmap)
    rps = regionprops(label_image)
    if len(rps) == 1:
        bounds = rps[0].bbox
    else:
        bounds = [np.inf, np.inf, np.inf, -1*np.inf, -1*np.inf, -1*np.inf]
        for ridx in range(len(rps)):
            selected_bounds = rps[ridx].bbox
            for bidx in [0,1,2]: #minimums
                if selected_bounds[bidx] < bounds[bidx]:
                    bounds[bidx] = selected_bounds[bidx]
            for bidx in [3,4,5]: #maximums
                if selected_bounds[bidx] > bounds[bidx]:
                    bounds[bidx] = selected_bounds[bidx]
    
    lef = bounds[5]; rig = bounds[2]
    ant = bounds[1]; pos = bounds[4]
    inf = bounds[3]; sup = bounds[0]
    return lef, rig, ant, pos, inf, sup