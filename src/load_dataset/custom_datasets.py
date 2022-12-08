#custom_datasets.py
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
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from src.load_dataset.vol_proc import mask, ctvol_preproc
from src.load_dataset.label_proc import organize_labels
from src.load_dataset.label_proc.vocab import vocabulary_ct

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

###################################################
# PACE Dataset for Data Stored in 2019-10-BigData #-----------------------------
###################################################
class CTDataset_2019_10(Dataset):    
    def __init__(self, setname, verbose, label_type_ld, genericize_lung_labels,
                 label_counts, view, crop_magnitude, use_projections9,
                 loss_string, volume_prep_args, attn_gr_truth_prep_args,
                 selected_note_acc_files, ct_scan_path,
                 ct_scan_projections_path, key_csvs_path, segmap_path):
        """CT Dataset class that works for preprocessed data in 2019-10-BigData.
        A single example (for crop_type == 'single') is a 4D CT volume:
            if num_channels == 3, shape [134,3,420,420]
            if num_channels == 1, shape [402,420,420]
        
        Variables:
        <setname> is either 'train' or 'valid' or 'test'
        <verbose>: if True then print output related to masks and save
            visualizations of masks (slow!)
        <label_type_ld> is either
            'disease' (for disease labels), or
            'location_disease_1218' (for location and disease flattened label vector)
        <genericize_lung_labels>: if True, then modify the location x disease
            labels so that the final ground truth includes only heart and
            generic lung labels, and you return additional keys in the sample
            for 'left_lung_gr_truth' and 'right_lung_gr_truth'. This is needed
            for models like BodyDiseaseSpatialAttn4Mask.
        <label_counts> is a dict with keys 'mincount_lung' and 'mincount_heart'
            and values that are ints which specify the minimum number of
            location x disease positive examples there must be in the 25,355
            training set scans for that label to be included.
        <view>: determines which axis comes first: axial, coronal, or sagittal.
            This matters because we are using models that have 2D feature
            extractors. <view> can also be 'all' in which case all 3 views
            will be returned.
        <crop_magnitude>: one of ['original','cube']
            'original' is what almost all my experiments use. This results in
                a CT volume cropped to [405,420,420]. This option MUST be
                chosen if <use_projections9> is True. This option also MUST
                be chosen if using a mask loss (because of the data aug
                implications, documented in ctvol_preproc.py)
            'cube' for a CT volume cropped to [405, 405, 405]. This is used
                in the text classifier experiments to enable full 360 degree
                flips and rotations so that text can be classified in any plane
                without having to train separate classifier for each plane.
        <use_projections9>: if True, then load the projections which have max
            pooling across 9 slices.
        <loss_string>: the loss string for the model. If the loss string indicates
            that this is a mask model, then an attention ground truth will be
            calculated.
        <volume_prep_args>: dict with keys pixel_bounds, num_channels, crop_type,
            selfsupervised, from_seg. For details of each of these args
            please see the documentation in ctvol_preproc.py for the function
            prepare_ctvol_2019_10_dataset(). Args that are added in __init__
            are max_slices and max_side_length
        <attn_gr_truth_prep_args>: key 'dilate' is True or False, and key
            'downsamp_algo' is a string specifying the algorithm to use for
            downsampling.
        
        Paths:
        Note that everywhere it says 'DEID' in the example paths, it could
        alternatively say 'PHI' if using the PHI version of the dataset.
        
        <selected_note_acc_files>: This should be a dictionary
            with key equal to setname and value that is a string. If the value
            is a path to a file, the file must be a CSV with one note
            accession per line. Only note accessions in this file will be used.
            If the value is not a valid file path, all available note accs
            will be used.
        <ct_scan_path>: path to the directory containing the CT scans saved
            as numpy arrays. e.g. /scratch/rlb61/2019-10-BigData-DEID
        <ct_scan_projections_path>: path to the directory containing the
            projected CT scans saved as numpy arrays.
            e.g. /scratch/rlb61/2020-04-15-Projections-DEID
        <key_csvs_path>: the path to a directory containing all of the
            important CSVs for the RAD-ChestCT data set.
            e.g. '/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/'
            CSVs that must be present:
            extrema_DEID.csv: the extrema CSV file which has the
                following columns: ['VolumeAcc_DEID', 'NoteAcc_DEID',
                'MRN_DEID','Set_Assigned','Subset_Assigned','sup_axis0min',
                'inf_axis0max','ant_axis1min','pos_axis1max','rig_axis2min',
                'lef_axis2max','shape0','shape1','shape2'].
                This file defines the data split, maps between volume accessions
                and note accessions, and defines lung extrema.
            Labels files that will be used in organize_labels.py, e.g.:
            labels/2020-03-23_duke_disease/imgtest_BinaryLabels_DEID.csv and .pkl
            labels/2020-03-23_duke_disease/imgtrain_BinaryLabels_DEID.csv and .pkl
            labels/2020-03-23_duke_disease/imgvalid_BinaryLabels_DEID.csv and .pkl
        <segmap_path>: the path to the directory containing the binary
            precomputed lung segmentation masks. Needed for mask models.
            e.g. /storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-segmaps-DEID/"""
        self.setname = setname
        assert setname in ['train','valid','test']
        self._define_subsets_list()
        self.verbose=verbose
        
        self.label_type_ld = label_type_ld
        assert isinstance(self.label_type_ld,str)
        
        self.genericize_lung_labels = genericize_lung_labels
        assert isinstance(self.genericize_lung_labels,bool)
        if self.genericize_lung_labels:
            #check that these are location_disease labels, because the
            #genericization only makes sense if we have both left lung and
            #right lung labels that we want to combine into generic 'lung' labels
            assert self.label_type_ld == 'location_disease_0323'
            self.disease_generic_label_df = organize_labels.read_in_labels('disease_0323', label_counts, setname, key_csvs_path)
        
        #View
        self.view = view
        assert view in ['axial','coronal','sagittal','all']
        if view in ['coronal','sagittal','all']:
            assert volume_prep_args['crop_type'] != 'triple', 'NotImplementedError: cannot have triple crops with a non-axial view'
        
        #Crop magnitude
        self.crop_magnitude = crop_magnitude
        assert self.crop_magnitude in ['original','cube']
        
        #Projections
        self.use_projections9 = use_projections9
        
        #Data augmentation
        data_augment_settings = {'train':True,'valid':False,'test':False}
        self.data_augment = data_augment_settings[self.setname]
        print('For',self.setname,'setting data_augment=',self.data_augment)
        
        #Determine whether you need to calculate an attention ground truth
        self.loss_string = loss_string
        if 'mask' in loss_string.lower():
            self.calculate_attn_gr_truth = True
            assert self.genericize_lung_labels #genericized lung labels are needed to make the attn_gr_truth
            assert self.crop_magnitude == 'original', 'Masks do NOT presently work if doing non-original crop_magnitude, because of the extra data augmentation for non-original crop_magnitudes'
        else:
            self.calculate_attn_gr_truth = False
        
        self.attn_gr_truth_prep_args = attn_gr_truth_prep_args
        self.selected_note_acc_files = selected_note_acc_files
        self.segmap_path = segmap_path
        
        #Define filenames and column names (DEID vs PHI)
        if 'DEID' in key_csvs_path:
            self.suffix = '_DEID.'
            self.noteacc_colname = 'NoteAcc_DEID'
        elif 'PHI' in key_csvs_path:
            self.suffix = '_PHI.'
            self.noteacc_colname = 'Accession'
        else:
            assert False, 'key_csvs_path must contain DEID or PHI to indicate which data naming scheme is being used'
        
        #Define paths
        if self.use_projections9:
            self.chosen_ct_scan_path = os.path.join(ct_scan_projections_path,view)
            print('\tLoading decompressed proj9 arrays from',self.chosen_ct_scan_path)
            self.load_format = '.npy'
        else:
            self.chosen_ct_scan_path = ct_scan_path
            print('Loading compressed arrays from',self.chosen_ct_scan_path)
            self.load_format = '.npz'
        
        #The extrema file defines the train/val/test split and the mapping
        #between volume accessions and note accessions.
        self.extrema = ctvol_preproc.load_extrema(os.path.join(key_csvs_path,'extrema.csv'.replace('.',self.suffix)))
        self.volume_accessions, self.note_accessions = self.get_accessions_this_set()
        
        #Get the ground truth labels
        labels_df = organize_labels.read_in_labels(self.label_type_ld, label_counts, self.setname, key_csvs_path)
        #Filter so that only volumes you are actually going to use are included
        self.labels_df = labels_df.filter(items=self.note_accessions,axis='index')
        assert self.labels_df.shape[0]==len(self.note_accessions)
        
        #Finalize volume prep args
        self.volume_prep_args = self.complete_volume_prep_args(volume_prep_args)
        self.print_arg_summary()
        self.complete_attn_gr_truth_prep_args()
    
    def complete_volume_prep_args(self, volume_prep_args):
        if self.crop_magnitude == 'original':
            max_slices = 405
            max_side_length = 420 
        elif self.crop_magnitude == 'cube':
            max_slices = 405
            max_side_length = 405
        
        #Set the CT volume shape
        volume_prep_args['max_side_length'] = max_side_length
        if self.use_projections9:
            assert self.crop_magnitude=='original', 'When use_projections9 is True, crop_magnitude must be original, not'+str(self.crop_magnitude)
            assert not volume_prep_args['from_seg'], 'Cannot use projections with lung segmentation cropping'
            volume_prep_args['max_slices'] = 45 #nine times smaller than 405
        else:
            volume_prep_args['max_slices'] = max_slices 
        #Checks
        if volume_prep_args['crop_type'] == 'triple':
            assert self.label_type_ld in ['location_disease_0323','location_disease_1218','heart','left_lung','right_lung']
        if volume_prep_args['selfsupervised']:
            #binary cross entropy loss ONLY for selfsupervised learning
            assert (('bce-selfsup-schedule' in self.loss_string) or (self.loss_string == 'bce'))
            assert volume_prep_args['crop_type']=='single'
        if self.calculate_attn_gr_truth:
            assert self.view == 'axial'
            assert volume_prep_args['crop_type'] == 'single'
            assert volume_prep_args['from_seg'] == False
        return volume_prep_args
    
    def print_arg_summary(self):
        print('For dataset',self.setname+':')
        summary = {'label_type_ld=':self.label_type_ld,
                   'view=':self.view}
        for key in list(self.volume_prep_args.keys()):
            summary[key] = self.volume_prep_args[key]
        for key in list(self.attn_gr_truth_prep_args):
            summary[key] = self.attn_gr_truth_prep_args[key]
        for key in summary.keys():
            print('\t',key,summary[key])
    
    def complete_attn_gr_truth_prep_args(self):
        """Add additional arguments needed for the attention ground truth
        creation"""
        self.attn_gr_truth_prep_args['verbose'] = self.verbose
        self.attn_gr_truth_prep_args['extrema'] = self.extrema
        self.attn_gr_truth_prep_args['max_slices'] = self.volume_prep_args['max_slices']
        self.attn_gr_truth_prep_args['max_side_length'] = self.volume_prep_args['max_side_length']
        self.attn_gr_truth_prep_args['use_projections9'] =  self.use_projections9
        self.attn_gr_truth_prep_args['n_outputs'] = len(self.return_label_meanings())
        self.attn_gr_truth_prep_args['segmap_path'] = self.segmap_path
    
    # Pytorch Required Methods #------------------------------------------------
    def __len__(self):
        return len(self.volume_accessions)
        
    def __getitem__(self, idx):
        """Return a single sample at index <idx>. The sample is a Python
        dictionary with the following keys:
        'data': the CT volume pixel data
        'gr_truth': the binary vector of abnormality ground truth
        'volume_acc': the volume accession number for this example
        'auglabel': a binary vector summarizing data flips and rotations
        'randpad6val': a vector summarizing random padding"""
        return self._get_pace(self.volume_accessions[idx], self.note_accessions[idx])
    
    # Obtaining Accession Numbers for this Set #--------------------------------
    def get_accessions_this_set(self):
        """Return the volume accession and note accession numbers for
        this set (train, val, or test)"""
        #Filter to select rows that are part of this subset (e.g. 'imgtrain')
        #Note that in self.extrema, the index is VolumeAcc_DEID and one of the
        #columns is NoteAcc_DEID
        selected_subset = self.extrema[self.extrema['Subset_Assigned'].isin(self.subsets_list)]
        
        #Shape checks
        if self.setname=='train':
            assert selected_subset.shape[0]==25355
        elif self.setname=='valid':
            assert selected_subset.shape[0]==2085
        elif self.setname=='test':
            assert selected_subset.shape[0]==7209
        
        #Filter to select examples that are part of the predefined subset,
        #if you are using a predefined subset:
        setname_file = self.selected_note_acc_files[self.setname]
        if 'predefined_subsets' in setname_file:
            assert os.path.isfile(setname_file)
            print('\tUsing predefined subset from',setname_file)
            sel_accs = pd.read_csv(setname_file,header=0)       
            assert sorted(list(set(sel_accs['Subset_Assigned'].values.tolist())))==sorted(self.subsets_list)
            selected_subset = selected_subset[selected_subset[self.noteacc_colname].isin(sel_accs.loc[:,self.noteacc_colname].values.tolist())]
            
            #Shape check:
            if self.setname=='train':
                assert selected_subset.shape[0]==2000
            elif self.setname=='valid':
                assert selected_subset.shape[0]==1000
            
        print('\tFinal total examples in requested subsets:',selected_subset.shape[0])
        #According to this thread: https://github.com/pytorch/pytorch/issues/13246
        #it is better to use a numpy array than a list to reduce memory leaks.
        #The only reason we're returning note_accessions too is so we can filter
        #the labels df to only include examples we are going to actually use,
        #for the use case of a weighted loss where the weights are based on the
        #fraction of positives. (We want to calculate the fraction of positives
        #based on the training set that is actually used)
        volume_accessions = selected_subset.index.values #e.g. array(['trn19675.npz', 'trn19706.npz', 'trn15370.npz', ...,'trn21610.npz', 'trn04863.npz', 'trn15565.npz'], dtype=object)
        note_accessions = selected_subset[self.noteacc_colname].values #e.g. array(['trn19675', 'trn19706', 'trn15370', ..., 'trn21610', 'trn04863','trn15565'], dtype=object)
        
        #In the deidentified data, which we are using now, the volume and note
        #accessions are the same except for the ending (this was not true in
        #the original PHI data)
        if 'DEID' in self.suffix:
            assert [x.replace('.npz','').replace('.npy','') for x in volume_accessions]==note_accessions.tolist()
        return volume_accessions, note_accessions
    
    # Fetch a CT Volume (__getitem__ implementation) #--------------------------
    def _get_pace(self, volume_acc, note_acc):
        """<volume_acc> is for example RHAA12345_6.npz
        Return a dictionary with keys 'data', 'gr_truth', 'volume_acc',
        and 'auglabel' (where auglabel describes the data augmentations
        that were performed, and is used in self supervised learning or for
        making a mask.)
        Format of value for 'data':
            (1) If self.view == 'axial' OR 'coronal' OR 'sagittal' then:
                    If self.crop_type == 'single': data is a single array
                    If self.crop_type == 'triple': data is a dictionary with keys
                        'right_lung', 'heart', 'left_lung' and each value is an array
        
            (2) If self.view == 'all' then: return a dictionary with keys
                'axial', 'coronal', and 'sagittal' where the value is
                an array (if crop type single) or a dict (if crop type triple)
                as described above."""
        assert note_acc in volume_acc #for DEID or PHI data this should be true
        
        #ctvol [slices, square, square]
        if not self.use_projections9: #Load raw axial arrays
            data, auglabel, randpad6val = self._prepare_views_original_volume(volume_acc)
        elif self.use_projections9: #Load pre-saved projected arrays
            data, auglabel, randpad6val = self._prepare_views_projections(volume_acc)
        gr_truth = self._prepare_this_grtruth(note_acc)
        sample = {'data': data, 'gr_truth': gr_truth, 'volume_acc': volume_acc,
                  'auglabel':auglabel, 'randpad6val':randpad6val}
        if self.genericize_lung_labels:
            sample = make_lung_labels_generic(sample,self.labels_df.columns.values.tolist(),self.disease_generic_label_df,note_acc)
        if self.calculate_attn_gr_truth:
            #Must calculate attention ground truth AFTER make_lung_labels_generic
            #because you need the organ-specific classification ground truths
            #in order to make the attention ground truth.
            sample['attn_gr_truth'] = mask.ConstructAttnGroundTruth(sample=sample,**self.attn_gr_truth_prep_args).attn_gr_truth
        return sample
    
    def _prepare_views_original_volume(self, volume_acc):
        """Prepare axial, coronal, sagittal, or all views based on the original
        raw data which is a whole, unprojected, array saved in compressed form"""
        assert self.load_format == '.npz' #Load compressed npz file
        ctvol = np.load(os.path.join(self.chosen_ct_scan_path, volume_acc))['ct']
        #Obtain the desired view. Default view is axial: [ax, cor, sag]
        if self.view == 'axial':
            return self._prepare_this_ctvol(ctvol, volume_acc)
        if self.view == 'coronal': #[cor, sag, ax]
            return self._prepare_this_ctvol( np.transpose(ctvol,[1,0,2]), volume_acc)
        elif self.view == 'sagittal': #[sag, ax, cor]
            return self._prepare_this_ctvol( np.transpose(ctvol,[2,0,1]), volume_acc)
        elif self.view == 'all':
            #auglabel not implemented. randpad6val not implemented.
            axial, _, _ = self._prepare_this_ctvol(copy.deepcopy(ctvol), volume_acc)
            coronal, _, _ = self._prepare_this_ctvol( np.transpose(copy.deepcopy(ctvol),[1,0,2]), volume_acc)
            sagittal, _, _ = self._prepare_this_ctvol( np.transpose(copy.deepcopy(ctvol),[2,0,1]), volume_acc)
            data = {'axial':axial,'coronal':coronal,'sagittal':sagittal}
            return data, 0, 0 #return 0 instead of None because batches can't contain None
    
    def _prepare_views_projections(self, volume_acc):
        """Prepare axial, coronal, sagittal, or all views based on the projected
        data which is an array upon which max pooling over 9 slices in the
        relevant direction was applied before saving as a decompressed array."""
        assert self.load_format == '.npy' #Load decompressed npy file
        if self.view != 'all':
            ctvol = np.load(os.path.join(self.chosen_ct_scan_path, volume_acc.replace('.npz','_'+self.view+'_proj9.npy')))
            return self._prepare_this_ctvol(ctvol, volume_acc)
        elif self.view == 'all':
            #auglabel not implemented. randpad6val not implemented.
            axial, _, _ = self._prepare_this_ctvol(np.load(os.path.join(self.chosen_ct_scan_path.replace('all','axial'), volume_acc.replace('.npz','_axial_proj9.npy'))), volume_acc)
            coronal, _, _ = self._prepare_this_ctvol(np.load(os.path.join(self.chosen_ct_scan_path.replace('all','coronal'), volume_acc.replace('.npz','_coronal_proj9.npy'))), volume_acc)
            sagittal, _, _ = self._prepare_this_ctvol(np.load(os.path.join(self.chosen_ct_scan_path.replace('all','sagittal'), volume_acc.replace('.npz','_sagittal_proj9.npy'))), volume_acc)
            data = {'axial':axial,'coronal':coronal,'sagittal':sagittal}
            return data, 0, 0 #return 0 instead of None because batches can't contain None
    
    def _prepare_this_ctvol(self, ctvol, volume_acc):
        """Major preprocessing on the <ctvol>"""
        return ctvol_preproc.prepare_ctvol_2019_10_dataset(ctvol = ctvol,
                                                   volume_acc = volume_acc,
                                                   extrema = self.extrema,
                                                   data_augment = self.data_augment,
                                                   **self.volume_prep_args)
    
    def _prepare_this_grtruth(self, note_acc):
        """Obtain the ground truth for this particular <volume_acc>"""
        gr_truth = self.labels_df.loc[note_acc, :].values
        gr_truth = torch.from_numpy(gr_truth).squeeze().type(torch.float)
        return gr_truth
    
    def return_label_meanings(self):
        """Return a list of strings that describe the labels. The strings
        must be in the same order that the ground truth is returned, e.g.
        if cardiomegaly is at index 0 in the ground truth vector then
        the string 'cardiomegaly' must appear at index 0 in the label
        meanings list."""
        #TODO make genericizing the lung labels something that happens once
        #to the entire dataframe not something that happens dynamically one
        #example at a time
        
        #The majority of the time, the label meanings can just be read directly
        #out of the labels_df:
        label_meanings = self.labels_df.columns.values.tolist()
        
        #If we are using generic lung labels, 'fold together' the right lung
        #and the left lung into just 'lung':
        if self.genericize_lung_labels:
            print('Genericizing the lung labels')
            n_outputs_heart, n_outputs_lung = infer_n_outputs_heart_and_lung(self.labels_df.columns.values.tolist())
            heart_label_meanings = label_meanings[0:n_outputs_heart]
            lung_label_meanings_derived_left = [x.replace('left_','') for x in label_meanings[n_outputs_heart:n_outputs_heart+n_outputs_lung]]
            lung_label_meanings_derived_right = [x.replace('right_','') for x in label_meanings[n_outputs_heart+n_outputs_lung:]]
            assert lung_label_meanings_derived_left == lung_label_meanings_derived_right
            print('\tHeart label meanings:',heart_label_meanings)
            print('\tLung label meanings:',lung_label_meanings_derived_right)
            label_meanings = heart_label_meanings + lung_label_meanings_derived_right
        
        #if self-supervised learning and label_meanings 'all' that means
        #we are going to train on disease labels and fliprot labels, so
        #add in the fliprot label names:
        if self.volume_prep_args['selfsupervised']:
            label_meanings = label_meanings + ['flip0','flip1','flip2','rot0','rot1','rot2']
        
        print('Final label meanings ('+str(len(label_meanings))+' labels total):',label_meanings)
        return label_meanings
    
    # Sanity Check #------------------------------------------------------------
    def _define_subsets_list(self):
        assert self.setname in ['train','valid','test']
        if self.setname == 'train':
            self.subsets_list = ['imgtrain']
        elif self.setname == 'valid':
            self.subsets_list = ['imgvalid_a']
        elif self.setname == 'test':
            self.subsets_list = ['imgtest_a','imgtest_b','imgtest_c','imgtest_d']
        print('Creating',self.setname,'dataset with subsets',self.subsets_list)

def make_lung_labels_generic(sample, label_meanings, disease_generic_label_df, note_acc):
    #Done with testing. See 2020-09-09-Updating-and-Testing-the-GenLung-Labels.docx
    """Return the <sample> so that it's been modified to contain
    a rearranged_gr_truth under the key 'gr_truth' and so that it also contains
    a 'heart_gr_truth', 'left_lung_gr_truth', and 'right_lung_gr_truth.'
    
    The rearranged_gr_truth is created as follows: Take the original <gr_truth>
    which contains heart, left lung, and right lung ground truth, and combine
    the left lung and right lung labels into generic lung labels.
    Because I sorted the lung labels lexicographically I can get the
    generic lung labels as follows:
    heart_thing1
    heart_thing2
    rightlung_thing3   ----\   rightlung_thing3 OR leftlung_thing3 = thing3
    rightlung_thing4   ----/   rightlung_thing4 OR leftlung_thing4 = thing4
    leftlung_thing3
    leftlung_thing4
    
    Note that <disease_generic_label_df> is the diseases dataframe of labels.
    It includes labels that are not specific to any location, i.e. they are
    for diseases only."""
    assert note_acc in sample['volume_acc'] #sanity check
    n_outputs_heart, n_outputs_lung = infer_n_outputs_heart_and_lung(label_meanings)
    gr_truth = sample['gr_truth'] #e.g. shape [132]
    
    #Rearrange the ground truth as described in function docstring. We know
    #that the labels appear in the order heart, left_lung, right_lung and that
    #the diseases for the lungs are in the same order for the right
    #lung and the left lung because the labels are in lexicographic order:
    heart_gr_truth = gr_truth[0:n_outputs_heart] #e.g. shape [30]
    left_lung_gr_truth = gr_truth[n_outputs_heart:n_outputs_heart+n_outputs_lung] #e.g. shape [51]
    left_lung_label_meanings = label_meanings[n_outputs_heart:n_outputs_heart+n_outputs_lung] #for sanity checks
    right_lung_gr_truth = gr_truth[n_outputs_heart+n_outputs_lung:] #e.g. shape [51]
    right_lung_label_meanings = label_meanings[n_outputs_heart+n_outputs_lung:] #for sanity checks
    assert [x.replace('left_lung_','') for x in left_lung_label_meanings]==[x.replace('right_lung_','') for x in right_lung_label_meanings] #sanity check
    #Note that you can only do logical operations like A | B (A or B) on
    #Byte Tensors, hence converstion to a Byte tensor. But then it needs
    #to be a float for the loss computation so it gets converted back to float.
    lung_gr_truth = (left_lung_gr_truth.byte() | right_lung_gr_truth.byte()).float() #e.g. shape [51]
    
    #If we are using the right lung vs. left lung labels as the final
    #classification labels, then we want to use them as-is, i.e. exactly as
    #they are produced by the label extraction code. However, if we are using
    #genericized lung labels, that means we are doing a model with a mask loss,
    #in which the right lung vs. left lung labels are used to determine where the
    #model is allowed to look. There are a small fraction of cases where we might
    #know that a lung label is present (e.g. pneumonia) but we don't know
    #which side it's on. In these cases, we want to "update" the right lung and
    #left lung labels so that they have a '1' for this generic lung label, so
    #that the model is allowed to look in either lung if we don't know
    #exactly which lung it's in. 
    #genlung_label_order is e.g. ['air_trapping', 'airspace_disease', 'aspiration',
    #'atelectasis', ...,'soft_tissue', 'staple', 'suture', 'transplant', 'tree_in_bud']
    lung_specific_pathology = vocabulary_ct.return_lung_specific_pathology()
    genlung_label_order = [x.replace('right_lung_','') for x in label_meanings[-1*n_outputs_lung:]]
    for idx in range(len(genlung_label_order)):
        genlung_label = genlung_label_order[idx] #.e.g 'air_trapping'
        
        #check if this is a lung-specific label. e.g. 'pneumonia' is lung
        #specific and 'nodule' is not. If it's lung specific then update
        #all the lung labels based on the labels in disease_generic_label_df.
        if genlung_label in lung_specific_pathology:
            if disease_generic_label_df.at[note_acc,genlung_label]==1:
                left_lung_gr_truth[idx] = 1
                right_lung_gr_truth[idx] = 1
                lung_gr_truth[idx] = 1
                #Sanity check:
                assert (genlung_label==left_lung_label_meanings[idx].replace('left_lung_','')==right_lung_label_meanings[idx].replace('right_lung_',''))
            
    #Now combine the lung labels with the heart labels to get the final gr truth:
    rearranged_gr_truth = torch.cat((heart_gr_truth, lung_gr_truth), dim=0) #e.g. shape [81]
    
    #Replace the gr_truth in the sample with the rearranged_gr_truth
    sample['gr_truth'] = rearranged_gr_truth
    
    #Store the heart, left lung, and right lung gr truth (for use in special
    #attention losses)
    sample['heart_gr_truth'] = heart_gr_truth
    sample['left_lung_gr_truth'] = left_lung_gr_truth
    sample['right_lung_gr_truth'] =  right_lung_gr_truth
    return sample

def infer_n_outputs_heart_and_lung(label_meanings_raw):
    """<label_meanings> is a list of strings describing the heart and lung
    labels. Infer the number of lung diseases (i.e. half the number of
    right lung + left lung labels) and the number of heart diseases."""
    #label_meanings should be the raw version i.e. derived from the column
    #headers of the labels dataframe. So it should contain 'right_lung' and
    #'left_lung' - it hasn't been genericized yet.
    assert 'right_lung' in ' '.join(label_meanings_raw)
    assert 'left_lung' in ' '.join(label_meanings_raw)
    #Calculate n_outputs_lung and n_outputs_heart
    n_outputs_lung = int(len([x for x in label_meanings_raw if 'lung' in x])/2.0)
    n_outputs_heart = len(label_meanings_raw) - (2*n_outputs_lung)
    return n_outputs_heart, n_outputs_lung