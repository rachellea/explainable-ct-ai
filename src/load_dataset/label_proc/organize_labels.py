#organize_labels.py
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
import pickle
import numpy as np
import pandas as pd

from src.load_dataset.label_proc.vocab import vocabulary_ct

#######################
# Ground Truth Labels #---------------------------------------------------------
#######################
HEART_LOCATIONS, HEART_PATHOLOGY = vocabulary_ct.return_heart_keys()
RIGHT_LUNG_LOCATIONS = ['right_upper','right_mid','right_lower','right_lung']
LEFT_LUNG_LOCATIONS = ['left_upper','left_mid','left_lower','left_lung']
LUNG_LOCATIONS = ['lung','airways']+RIGHT_LUNG_LOCATIONS+LEFT_LUNG_LOCATIONS
_, LUNG_PATHOLOGY = vocabulary_ct.return_lung_keys()
GREAT_VESSEL_LOCATIONS, GREAT_VESSEL_PATHOLOGY = vocabulary_ct.return_great_vessel_keys()
MEDIASTINUM_LOCATIONS = ['mediastinum','hilum']
MEDIASTINUM_PATHOLOGY = ['calcification','cancer','lesion','lymphadenopathy',
                        'mass','nodule','nodulegr1cm','opacity']

def read_in_labels(label_type_ld, label_counts, setname, key_csvs_path):
    """Return a pandas dataframe with the dataset labels.
    Accession numbers are the index and labels (e.g. 'pneumonia', e.g.
    'heart_cardiomegaly') are the columns.
    <setname> can be 'train', 'valid', or 'test'.
    
    The format of <label_type_ld> describes, first, the locations and diseases
    that should be included; if it's location_disease then locations and
    diseases from the heart, mediastinum, and lungs will be included. If
    it's heart, left_lung, or right_lung then the locations and diseases
    will be restricted to that organ."""
    assert label_type_ld in ['disease_1218','disease_0323','location_disease_1218',
            'location_disease_0323','heart_0323','left_lung_0323','right_lung_0323']
    
    key_csvs_path_labels = os.path.join(key_csvs_path,'labels')
    if 'DEID' in key_csvs_path:
        suffix = '_DEID.'
    elif 'PHI' in key_csvs_path:
        suffix = '_PHI.'
    
    #This is a weird one because you can just read the labels file directly:.
    #I'm including this for legacy purposes. These are the labels I used for
    #my first paper.
    if label_type_ld == 'disease_1218':
        binlabels_path = os.path.join(os.path.join(key_csvs_path_labels,'2019-12-18_duke_disease'),'img'+setname+'_BinaryLabels.csv'.replace('.',suffix))
        return pd.read_csv(binlabels_path, header=0, index_col = 0)
    
    elif label_type_ld == 'disease_0323':
        binlabels_path = os.path.join(os.path.join(key_csvs_path_labels,'2020-03-23_duke_disease'),'img'+setname+'_BinaryLabels.csv'.replace('.',suffix))
        return pd.read_csv(binlabels_path, header=0, index_col = 0)
    
    #For all of these, the labels must be constructed:
    else:
        #Choose what file to load from, based on the date, 12-18 or 03-23
        if '1218' in label_type_ld:
            binlabels_path = os.path.join(os.path.join(key_csvs_path_labels,'2019-12-18_duke_disease'),'img'+setname+'_BinaryLabels.pkl'.replace('.',suffix))
        elif '0323' in label_type_ld:
            binlabels_path = os.path.join(os.path.join(key_csvs_path_labels,'2020-03-23_duke_disease'),'img'+setname+'_BinaryLabels.pkl'.replace('.',suffix))
        
        #Choose what locations and diseases to load:
        chosen_pairs = return_pairs(label_type_ld)
        
        #Print
        print('\tlabel_type_ld:',label_type_ld)
        print('\tLoading labels from:',binlabels_path)
        print('\tmincount_lung:',label_counts['mincount_lung'])
        print('\tmincount_heart:',label_counts['mincount_heart'])
        
        #Organize labels and return
        return make_custom_labels_df(chosen_pairs, binlabels_path, label_type_ld, label_counts['mincount_lung'], label_counts['mincount_heart'])

def return_pairs(label_type_ld):
    """Return 'pairs' which are formatted as a dict where the keys of the dict
    are location strings like 'h_great_vessel' or 'right_lung' and each value
    is a list of lists. A value is a list containing a list of locations
    and then a list of abnormalities.
    e.g. pairs = {'left_lung':[LEFT_LUNG_LOCATIONS,LUNG_PATHOLOGY],
                'right_lung':[RIGHT_LUNG_LOCATIONS,LUNG_PATHOLOGY]}"""
    global HEART_LOCATIONS, HEART_PATHOLOGY
    global RIGHT_LUNG_LOCATIONS, LEFT_LUNG_LOCATIONS, LUNG_PATHOLOGY
    global GREAT_VESSEL_LOCATIONS, GREAT_VESSEL_PATHOLOGY
    global MEDIASTINUM_LOCATIONS, MEDIASTINUM_PATHOLOGY
    
    pair_type = '_'.join(label_type_ld.split('_')[0:-1]) #e.g. in 'location_disease_1218' -> out 'location_disease'
    
    if pair_type == 'location_disease':
        #Note: the 'h' prefix is a hacky thing to force great_vessel, heart, and
        #mediastinum to all appear contiguously in the output labels
        #since the labels are sorted lexicographically.
        pairs = {'h_great_vessel':[GREAT_VESSEL_LOCATIONS,GREAT_VESSEL_PATHOLOGY],
            'h_mediastinum':[MEDIASTINUM_LOCATIONS,MEDIASTINUM_PATHOLOGY],
            'heart':[HEART_LOCATIONS,HEART_PATHOLOGY],
            'left_lung':[LEFT_LUNG_LOCATIONS,LUNG_PATHOLOGY],
            'right_lung':[RIGHT_LUNG_LOCATIONS,LUNG_PATHOLOGY]}
        return pairs
    
    elif pair_type == 'heart':
        pairs_heart = {'h_great_vessel':[GREAT_VESSEL_LOCATIONS,GREAT_VESSEL_PATHOLOGY],
            'h_mediastinum':[MEDIASTINUM_LOCATIONS,MEDIASTINUM_PATHOLOGY],
            'heart':[HEART_LOCATIONS,HEART_PATHOLOGY]}
        return pairs_heart
    
    elif pair_type == 'left_lung':
        pairs_left_lung = {'left_lung':[LEFT_LUNG_LOCATIONS,LUNG_PATHOLOGY]}
        return pairs_left_lung
    
    elif pair_type == 'right_lung':
       pairs_right_lung = {'right_lung':[RIGHT_LUNG_LOCATIONS,LUNG_PATHOLOGY]}
       return pairs_right_lung

def make_custom_labels_df(pairs, binlabels_path, label_type_ld, mincount_lung, mincount_heart): #Done with testing
    """Return a labels df.
    Variables:
    <pairs> is a dict where keys are organs and values are lists. The first
        element of the list is a locations list and the second element of
        the list is a diseases list."""
    #Make the labels dataframe with accessions as index and labels as columns
    labels_dict = pickle.load(open(os.path.abspath(binlabels_path),'rb'))
    accessions = list(labels_dict.keys())
    label_name_list = _return_label_name_list(pairs, binlabels_path, label_type_ld, mincount_lung, mincount_heart) #allowed labels
    print('\tlabel_name_list includes',len(label_name_list),'labels')
    labels_df = pd.DataFrame(np.zeros((len(accessions),len(label_name_list))),
                             index=accessions,
                             columns = label_name_list)
    for accession in accessions:
        #acc_df is a dataframe with locations as columns and diseases as index
        #which contains the labels for a particular accession number
        acc_df = labels_dict[accession]
        for organ in pairs.keys():
            locations = pairs[organ][0]
            diseases = pairs[organ][1]
            for loc in locations:
                for dis in diseases:
                    label = organ+'_'+dis #e.g. heart_cardiomegaly or lung_pneumonia
                    if label in label_name_list:
                        #The following 'if' is only needed because when testing
                        #we use tiny fake data that is missing some dis and some loc
                        if ((loc in acc_df.columns.values.tolist()) and (dis in acc_df.index.values.tolist())):
                            #Fill in the label:
                            if acc_df.at[dis,loc] == 1:
                                labels_df.at[accession,label] = 1
    return labels_df
    
def _return_label_name_list(pairs, binlabels_path, label_type_ld, mincount_lung, mincount_heart): #Done with testing
    """Return the final list of location x disease label strings formatted as
    location_disease e.g. ['right_lung_atelectasis', 'heart_cardiomegaly'].
    Only include location x disease labels that occur more than <mincount>
    times in the training set."""
    if label_type_ld in ['location_disease_1218','location_disease_0323',
                         'location_disease_9999','left_lung_9999']:
        #note that the 9999 ending is used for testing purposes.
        temp_label_type_ld = label_type_ld
    elif label_type_ld in ['heart_0323','left_lung_0323','right_lung_0323']:
        #Note: even for organ-specific labels, we ALWAYS calculate
        #traincount_flat_filt_keep using 'location_disease'. Why? Because we
        #want to make sure that only left lung and right lung diseases are
        #included which have at least <mincount_lung> examples in BOTH lungs.
        #If we load only the left_lung by itself, then we will not be able to
        #filter out left lung diseases which have an insufficient count in the
        #right lung.
        temp_label_type_ld = 'location_disease_0323'
    else:
        assert False, 'NOT IMPLEMENTED'
    traincount_flat_filt_keep = _load_traincount_flat_filt_keep(pairs, binlabels_path, temp_label_type_ld, mincount_lung, mincount_heart)
    keep = traincount_flat_filt_keep[traincount_flat_filt_keep['Decision']=='keep']
    label_name_list = keep['Label'].values.tolist()
    
    #Because for the heart, left lung, and right lung we actually loaded
    #all locations and diseases, here is where we apply the final desired
    #organ-specific label_type_ld by removing anything that is not in the organ
    #of interest:
    if label_type_ld in ['heart_0323','left_lung_0323','right_lung_0323']:
        pairs = return_pairs(label_type_ld)
        def contains_any(string, allowed):
            """Return True if <string> has as a substring any string in
            the list <allowed>"""
            for a in allowed:
                if a in string:
                    return True
            return False
        #allowed_strings for left_lung_0323 is ['left_lung'];
        #allowed_strings for heart is ['h_great_vessel','h_mediastinum','heart']
        allowed = list(pairs.keys())
        label_name_list = [x for x in label_name_list if contains_any(x,allowed)]
    return sorted(label_name_list)

def _load_traincount_flat_filt_keep(pairs, binlabels_path, label_type_ld, mincount_lung, mincount_heart):
    """If traincount_flat_filt_keep.csv already exists, then read it in.
    Otherwise, create it from scratch. The reason it's not created from
    scratch every time is that iterating over every single location x disease
    dataframe in the full training set of 25,355 examples takes a long time."""
    #tcffk stands for traincount_flat_filt_keep
    #example tcffk_path = './load_dataset/ground_truth/2020-03-23_duke_disease/location_disease_0323_tcffk_125_200.csv'
    tcffk_path = os.path.join(os.path.split(binlabels_path)[0],label_type_ld+'_tcffk_'+str(mincount_lung)+'_'+str(mincount_heart)+'.csv')
    if os.path.exists(tcffk_path):
        print('\tLoading tcffk from:',tcffk_path)
        traincount_flat_filt_keep = pd.read_csv(tcffk_path,header=0,index_col=0)
    else:
        print('\tCreating new tcffk which will be saved to:',tcffk_path)
        traincount_flat_filt_keep = _return_traincount_flat_filt_keep(pairs, tcffk_path, mincount_lung, mincount_heart)
    return traincount_flat_filt_keep

def _return_traincount_flat_filt_keep(pairs, tcffk_path, mincount_lung, mincount_heart): #Done with testing
    """Take the traincount_flat returned by _return_traincount_flat()
    and annotate with a column 'Decision' which is equal to 'keep' if the
    'TrainCount' exceeds the mincount for that organ, and is equal to 'exclude'
    otherwise.
    This function returns a df <traincount_flat_filt_keep> where the index contains
    strings that describe the location x disease labels (e.g. 'right_lung_nodule')
    and the columns are 'TrainCount' and 'Decision.' 'TrainCount' is an
    integer representing the count of the number of times that location x
    disease label is positive in the training set. 'Decision' is 'keep' for
    labels that are frequent enough to use in the model."""
    traincount_flat_filt_keep = _return_traincount_flat_filt(tcffk_path, pairs)
    
    #Filter by mincount, which is the minimum number of examples in the
    #training set that must be present for a particular location x disease
    #label to be included
    #idx is the locdis label e.g. 'right_lung_atelectasis'
    #Note that the right lung and left lung need to have the same labels so if
    #the count is too small for either lung, that label is excluded
    traincount_flat_filt_keep['Decision']='keep'
    for idx in traincount_flat_filt_keep.index.values.tolist():
        if 'lung' in idx:
            mincount = mincount_lung
        else:
            mincount = mincount_heart
        if traincount_flat_filt_keep.at[idx,'TrainCount'] <  mincount:
            traincount_flat_filt_keep.at[idx,'Decision']='exclude'
            if 'right_lung' in idx:
                left_idx = idx.replace('right_lung','left_lung')
                #we need to check whether left_idx is already in the index; it
                #will NOT be if for example we are focused on only the right
                #lung. We don't want to end up 'adding' left_idx to the index
                #if it's not already there!
                if left_idx in traincount_flat_filt_keep.index.values.tolist(): 
                    traincount_flat_filt_keep.at[left_idx,'Decision']='exclude'
            if 'left_lung' in idx:
                right_idx = idx.replace('left_lung','right_lung')
                if right_idx in traincount_flat_filt_keep.index.values.tolist():
                    traincount_flat_filt_keep.at[right_idx,'Decision']='exclude'
    print('\tLabels excluded because below',mincount,'count:',traincount_flat_filt_keep[traincount_flat_filt_keep['Decision']=='exclude'].shape[0])
    print('\tLabels kept:',traincount_flat_filt_keep[traincount_flat_filt_keep['Decision']=='keep'].shape[0])
    traincount_flat_filt_keep.to_csv(tcffk_path)
    return traincount_flat_filt_keep

def _return_traincount_flat_filt(tcffk_path, pairs): #Done with testing
    """Given the output of _return_traincount_flat(), if there are lungs
    involved, filter it so that each generic label is represented only once.
    e.g. the following input:
                        Label              TrainCount
    right_upper_mass    right_lung_mass       22
    right_mid_mass      right_lung_mass       16
    right_lower_mass    right_lung_mass       5
    right_lung_mass     right_lung_mass       33
    
    is filtered and then returned as:
                       TrainCount
    right_lung_mass       33"""
    traincount_flat = _return_traincount_flat(tcffk_path, pairs)
    print('\tTotal possible locdis labels before removing lung dups:',traincount_flat.shape[0])
    traincount_flat_filt = pd.DataFrame(columns=['Label','TrainCount'])
    #A single label, e.g. 'right_lung_nodule' may be associated with many
    #index detailed labels, e.g. 'right_upper_nodule', 'right_mid_nodule',
    #'right_lower_nodule', and 'right_lung_nodule.' In this case we want to
    #choose the index detailed label with the highest TrainCount (which
    #conceptually should be 'right_lung_nodule' in this example unless something
    #went really wrong with the original label creation)
    uniq_labels = list(set(traincount_flat['Label'].values.tolist()))
    for label in uniq_labels:
        sel = traincount_flat[traincount_flat['Label']==label]
        sel = sel.sort_values(by='TrainCount',ascending=False)
        #Get highest TrainCount and then extract all detailed labels with that
        #TrainCount, since there may be more than one detailed label with the
        #same TrainCount:
        top_TrainCount = sel.iat[0,1]
        sel_top = sel[sel['TrainCount']==top_TrainCount]
        if sel_top.shape[0]>1:
            #Check whether this is a lung label, since right now advanced filtering
            #is only implemented for lungs:
            lung_label = False
            for loc in ['lung','right_upper','right_mid','right_lower',
                        'left_upper','left_mid','left_lower']:
                if loc in label:
                    lung_label = True
            if lung_label:
                sel_top = _filter_sel_top_by_lung(sel_top)
            else: #then it's great vessel (e.g. ivc vs svc vs aorta) or heart
                #(e.g. heart vs mitral valve vs aortic valve) or some other
                #location with multiple sub-locations. In this case, we'll
                #just arbitrarily pick one of the sub-locations, e.g. if
                #sel_top is:
                #                               Label             TrainCount
                #ivc_staple               h_great_vessel_staple       1
                #pulmonary_artery_staple  h_great_vessel_staple       1
                #Then we can pick either ivc_staple or pulmonary_artery_staple
                #because the point is we just want to have a count of '1' for
                #h_great_vessel_staple
                sel_top = sel_top.sample(n=1)
        detailed_label = sel_top.index.values.tolist()[0]
        traincount_flat_filt.at[detailed_label,'Label'] = label
        traincount_flat_filt.at[detailed_label,'TrainCount'] = sel_top.at[detailed_label,'TrainCount']
    assert traincount_flat_filt.shape[0]==len(uniq_labels)
    traincount_flat_filt = traincount_flat_filt.sort_values(by=['TrainCount','Label'],ascending=False)
    print('\tTotal possible locdis labels after removing lung dups:',traincount_flat_filt.shape[0])
    return traincount_flat_filt

def _filter_sel_top_by_lung(sel_top):
    """Given a dataframe where the index is detailed label and the columns
    are Label and TrainCount, and all entries have the same Label and the same
    TrainCount, filter to keep only the detailed label that is biggest/
    most encompassing, based on lung anatomy -e.g. keep 'right_lung' over any
    lobes, or keep 'lung' over 'right_lung' or 'left_lung'
    For the lungs, the Label specifies which detailed label is biggest/most
    encompassing. Lungs need special handling because there are some lung
    labels which are explicit subsets of other lung labels (like right_upper,
    right_mid, and right_lower are subsets of right_lung, and so based on
    anatomy we know we want to pick right_lung in that case. This is different
    from say great vessels where the aorta is different from the svc or the ivc)"""
    assert len(set(sel_top['Label'].values.tolist()))==1
    return sel_top.filter(items=list(set(sel_top['Label'].values.tolist())),
                          axis='index')

def _return_traincount_flat(tcffk_path, pairs): #Done with testing
    """Take the matrix (rectangular dataframe) returned by
    _return_traincount_matrix() and flatten it so that the index includes
    location x disease labels like 'heart_cardiomegaly' and
    'right_upper_pneumonia' and there is one column called 'TrainCount' which
    contains the corresponding location x disease training set count read
    out of the traincount matrix, and another column called 'Label' which
    has the genericized label e.g. right_upper_pneumonia becomes right_lung_pneumonia"""
    #Collect training set counts for desired location x disease labels
    traincount = _return_traincount_matrix(tcffk_path)
    
    #Collect label names (e.g. 'right_lung_pneumonia') and counts
    traincount_flat = pd.DataFrame(columns=['Label','TrainCount'])
    for organ in pairs.keys():
        locations = pairs[organ][0]
        diseases = pairs[organ][1]
        for loc in locations:
            for dis in diseases:
                if ((loc in traincount.columns.values.tolist()) and (dis in traincount.index.values.tolist())):
                    detailed_label = loc+'_'+dis #e.g. right_upper_mass, heart_cardiomegaly, or left_lower_pneumonia
                    label = organ+'_'+dis #e.g. right_lung_mass, heart_cardiomegaly, or left_lung_pneumonia
                    #note that the detailed labels 'right_upper_mass',
                    #'right_mid_mass', and 'right_lower_mass' can all map to the
                    #label 'right_lung_mass' because they all share the organ 'right_lung'.
                    #That is why the index is the detailed_label rather than the
                    #generic label.
                    traincount_flat.at[detailed_label,'Label'] = label
                    traincount_flat.at[detailed_label,'TrainCount'] = traincount.at[dis,loc]
                else:
                    #The check for whether loc and dis are in the columns
                    #and index respectively is needed because when testing this
                    #code I use small fake data that doesn't include all
                    #possible locations and diseases.
                    #I raise an error if I'm using real data (i.e. data without
                    #9999 in the path) and I don't find a particular loc dis pair
                    if '9999' not in tcffk_path: assert False, 'WARNING: '+loc+' '+dis+' NOT FOUND'
    traincount_flat = traincount_flat.sort_values(by = ['TrainCount','Label'],ascending=False)
    return traincount_flat

def _return_traincount_matrix(tcffk_path): #Done with testing
    """Return a pandas dataframe with index of diseases and columns of locations
    where the entries are the count of that location x disease in the training set"""
    #e.g. tcffk_path = './load_dataset/ground_truth/2019-12-18_duke_disease/location_disease_0323_tcffk_125_200.csv'
    #>>> os.path.split(tcffk_path)
    #('./load_dataset/ground_truth/2019-12-18_duke_disease', 'traincount_flat_filt_keep.csv')
    if 'DEID' in tcffk_path:
        suffix = '_DEID.'
    elif 'PHI' in tcffk_path:
        suffix = '_PHI.'
    train_path = os.path.join(os.path.split(tcffk_path)[0],'imgtrain_BinaryLabels.pkl'.replace('.',suffix))
    print('\tLoading training set data from',train_path)
    labels_dict = pickle.load(open(train_path,'rb'))
    temp = labels_dict[list(labels_dict.keys())[0]]
    heatmap = np.zeros(temp.shape)
    for key in list(labels_dict.keys()):
        heatmap += labels_dict[key].values
    return pd.DataFrame(heatmap, columns = temp.columns.values.tolist(), index = temp.index.values.tolist())
