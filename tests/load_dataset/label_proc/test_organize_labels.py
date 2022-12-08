#test_organize_labels.py
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
import torch
import shutil
import pickle
import unittest
import pandas as pd

from tests import equality_checks as eqc
from src.load_dataset import custom_datasets
from src.load_dataset.label_proc import organize_labels

########################################
# Functions for constructing fake data #----------------------------------------
########################################
def return_fake_abn_and_loc_lists():
    """Return a list of abnormality strings, and a list of location strings,
    which will be used as the index and column labels respectively in the
    fake dataframes"""
    #pick abnormalities which are organ specific, and other abnormalities
    #which can be found in any organ
    abn = ['groundglass','honeycombing','atelectasis','nodule','mass','cancer','cardiomegaly','heart_failure']
    loc = ['right_upper','right_mid','right_lung','left_upper','left_lower','left_lung','lung','aorta','mediastinum','heart']
    return abn, loc

def return_fake_imgtrain_BinaryLabels():
    # >>> x['AAFAKE12345']
    #                     left_upper  left_mid  ...  pancreas  other_location
    # bandlike_or_linear         0.0       0.0  ...       0.0             0.0
    # groundglass                0.0       0.0  ...       0.0             0.0
    # honeycombing               0.0       0.0  ...       0.0             0.0
    # reticulation               0.0       0.0  ...       0.0             0.0
    # tree_in_bud                0.0       0.0  ...       0.0             0.0
    # ...                        ...       ...  ...       ...             ...
    # transplant                 0.0       0.0  ...       0.0             0.0
    # chest_tube                 0.0       0.0  ...       0.0             0.0
    # tracheal_tube              0.0       0.0  ...       0.0             0.0
    # gi_tube                    0.0       0.0  ...       0.0             0.0
    # other_path                 0.0       0.0  ...       0.0             0.0
    #
    # >>> x['AAFAKE12345'].columns.values.tolist()
    # ['left_upper', 'left_mid', 'left_lower', 'right_upper', 'right_mid', 'right_lower',
    #'right_lung', 'left_lung', 'lung', 'airways', 'heart', 'mitral_valve', 'aortic_valve',
    #'tricuspid_valve', 'pulmonary_valve', 'aorta', 'svc', 'ivc', 'pulmonary_artery',
    #'pulmonary_vein', 'right', 'left', 'anterior', 'posterior', 'superior', 'inferior',
    #'medial', 'lateral', 'interstitial', 'subpleural', 'centrilobular', 'thyroid',
    #'breast', 'axilla', 'chest_wall', 'rib', 'spine', 'bone', 'mediastinum', 'diaphragm',
    #'hilum', 'abdomen', 'esophagus', 'stomach', 'intestine', 'liver', 'gallbladder',
    #'kidney', 'adrenal_gland', 'spleen', 'pancreas', 'other_location']
    # >>> x['AAFAKE12345'].index.values.tolist()  
    # ['bandlike_or_linear', 'groundglass', 'honeycombing', 'reticulation', 'tree_in_bud',
    #'airspace_disease', 'air_trapping', 'aspiration', 'atelectasis', 'bronchial_wall_thickening',
    #'bronchiectasis', 'bronchiolectasis', 'bronchiolitis', 'bronchitis', 'emphysema',
    #'hemothorax', 'interstitial_lung_disease', 'lung_resection', 'mucous_plugging',
    #'pleural_effusion', 'pleural_thickening', 'pneumonia', 'pneumonitis', 'pneumothorax',
    #'pulmonary_edema', 'septal_thickening', 'tuberculosis', 'cabg', 'cardiomegaly',
    #'coronary_artery_disease', 'heart_failure', 'heart_valve_replacement',
    #'pacemaker_or_defib', 'pericardial_effusion', 'pericardial_thickening',
    #'sternotomy', 'arthritis', 'atherosclerosis', 'aneurysm', 'breast_implant',
    #'breast_surgery', 'calcification', 'cancer', 'catheter_or_port', 'cavitation',
    #'clip', 'congestion', 'consolidation', 'cyst', 'debris', 'deformity',
    #'density', 'dilation_or_ectasia', 'distention', 'fibrosis', 'fracture',
    #'granuloma', 'hardware', 'hernia', 'infection', 'infiltrate', 'inflammation',
    #'lesion', 'lucency', 'lymphadenopathy', 'mass', 'nodule', 'nodulegr1cm', 'opacity',
    #'plaque', 'postsurgical', 'scarring', 'scattered_calc', 'scattered_nod', 'secretion',
    #'soft_tissue', 'staple', 'stent', 'suture', 'transplant', 'chest_tube',
    #'tracheal_tube', 'gi_tube', 'other_path']
    abn, loc = return_fake_abn_and_loc_lists()
    #See 2020-08-06 word document for tables colorized
    patient_123 = pd.DataFrame([[0.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0],
                                [0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                [0.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],index=abn, columns=loc)
    patient_456 = pd.DataFrame([[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0],
                                [1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],
                                [0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]],index=abn, columns=loc)
    patient_789 = pd.DataFrame([[0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                [0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,0.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]],index=abn, columns=loc)
    imgtrain_BinaryLabels = {'patient_123':patient_123,
                             'patient_456':patient_456,
                             'patient_789':patient_789}
    return imgtrain_BinaryLabels

def init_testing_dir():
    """Must init a testing dir before running any tests;
    this testing dir contains the fake data"""
    if not os.path.exists('testing_delthis_DEID'):
        os.mkdir('testing_delthis_DEID')
    imgtrain_BinaryLabels = return_fake_imgtrain_BinaryLabels()
    pickle.dump(imgtrain_BinaryLabels, open(os.path.join('testing_delthis_DEID','imgtrain_BinaryLabels_DEID.pkl'), 'wb'))

##############
# Unit tests #------------------------------------------------------------------
##############
#correct answer pieces shared by test_return_traincount_flat_filt()
#and test_return_traincount_flat_filt_keep():
CORRECT_INDEX_FLAT_FILT = ['right_lung_groundglass','right_lung_honeycombing',
            'right_lung_atelectasis','right_lung_nodule','right_lung_mass',
            'right_lung_cancer','left_lung_groundglass','left_lung_honeycombing',
            'left_lung_atelectasis','left_lung_nodule','left_lung_mass',
            'left_lung_cancer','mediastinum_nodule','mediastinum_mass',
            'mediastinum_cancer','heart_cardiomegaly','heart_heart_failure']

CORRECT_DATA_FLAT_FILT = [['right_lung_groundglass',1.0],
                    ['right_lung_honeycombing',1.0],
                    ['right_lung_atelectasis',1.0],
                    ['right_lung_nodule',0.0],
                    ['right_lung_mass',0.0],
                    ['right_lung_cancer',0.0],
                    ['left_lung_groundglass',2.0],
                    ['left_lung_honeycombing',0.0],
                    ['left_lung_atelectasis',1.0],
                    ['left_lung_nodule',2.0],
                    ['left_lung_mass',1.0],
                    ['left_lung_cancer',0.0],
                    ['h_mediastinum_nodule',2.0],
                    ['h_mediastinum_mass',3.0],
                    ['h_mediastinum_cancer',2.0],
                    ['heart_cardiomegaly',2.0],
                    ['heart_heart_failure',2.0]]

class TestOrganizeLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_testing_dir()
    
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('testing_delthis_DEID')
    
    def test_return_traincount_matrix(self):
        tcffk_path = os.path.join('testing_delthis_DEID','9999_tcffk_num_num_DEID.csv')
        #return traincount produces a dataframe with abnormalities as the index
        #and locations as the columns, where the entries are the count of that
        #abnormality x location in the training set
        output = organize_labels._return_traincount_matrix(tcffk_path)
        abn, loc = return_fake_abn_and_loc_lists()
        correct = pd.DataFrame([[0.0,1.0,1.0,0.0,1.0,2.0,3.0,0.0,0.0,0.0],
                                [0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,0.0,0.0],
                                [1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0],
                                [0.0,0.0,0.0,2.0,1.0,2.0,2.0,0.0,2.0,0.0],
                                [0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,3.0,1.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,2.0,0.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0],
                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0]],index=abn, columns=loc)
        assert eqc.dfs_equal(output,correct)
        print('Passed test_return_traincount_matrix()')
    
    def test_return_traincount_flat(self):
        for label_type_ld in ['location_disease_9999','left_lung_9999']:
            tcffk_path = os.path.join('testing_delthis_DEID','9999_tcffk_num_num_DEID.csv')
            pairs = organize_labels.return_pairs(label_type_ld)
            output = organize_labels._return_traincount_flat(tcffk_path, pairs)
            #Define correct answer
            correct_index = ['right_upper_groundglass','right_upper_honeycombing','right_upper_atelectasis','right_upper_nodule',
                    'right_upper_mass','right_upper_cancer','right_mid_groundglass','right_mid_honeycombing',
                    'right_mid_atelectasis','right_mid_nodule','right_mid_mass','right_mid_cancer',
                    'right_lung_groundglass','right_lung_honeycombing','right_lung_atelectasis',
                    'right_lung_nodule','right_lung_mass','right_lung_cancer','left_upper_groundglass',
                    'left_upper_honeycombing','left_upper_atelectasis','left_upper_nodule',
                    'left_upper_mass','left_upper_cancer','left_lower_groundglass',
                    'left_lower_honeycombing','left_lower_atelectasis','left_lower_nodule',
                    'left_lower_mass','left_lower_cancer','left_lung_groundglass','left_lung_honeycombing',
                    'left_lung_atelectasis','left_lung_nodule','left_lung_mass','left_lung_cancer',
                    'mediastinum_nodule','mediastinum_mass','mediastinum_cancer',
                    'heart_cardiomegaly','heart_heart_failure']
            correct_data = [['right_lung_groundglass',0.0],
                            ['right_lung_honeycombing',0.0],
                            ['right_lung_atelectasis',1.0],
                            ['right_lung_nodule',0.0],
                            ['right_lung_mass',0.0],
                            ['right_lung_cancer',0.0],
                            ['right_lung_groundglass',1.0],
                            ['right_lung_honeycombing',0.0],
                            ['right_lung_atelectasis',1.0],
                            ['right_lung_nodule',0.0],
                            ['right_lung_mass',0.0],
                            ['right_lung_cancer',0.0],
                            ['right_lung_groundglass',1.0],
                            ['right_lung_honeycombing',1.0],
                            ['right_lung_atelectasis',1.0],
                            ['right_lung_nodule',0.0],
                            ['right_lung_mass',0.0],
                            ['right_lung_cancer',0.0],
                            ['left_lung_groundglass',0.0],
                            ['left_lung_honeycombing',0.0],
                            ['left_lung_atelectasis',1.0],
                            ['left_lung_nodule',2.0],
                            ['left_lung_mass',0.0],
                            ['left_lung_cancer',0.0],
                            ['left_lung_groundglass',1.0],
                            ['left_lung_honeycombing',0.0],
                            ['left_lung_atelectasis',1.0],
                            ['left_lung_nodule',1.0],
                            ['left_lung_mass',1.0],
                            ['left_lung_cancer',0.0],
                            ['left_lung_groundglass',2.0],
                            ['left_lung_honeycombing',0.0],
                            ['left_lung_atelectasis',1.0],
                            ['left_lung_nodule',2.0],
                            ['left_lung_mass',1.0],
                            ['left_lung_cancer',0.0],
                            ['h_mediastinum_nodule',2.0],
                            ['h_mediastinum_mass',3.0],
                            ['h_mediastinum_cancer',2.0],
                            ['heart_cardiomegaly',2.0],
                            ['heart_heart_failure',2.0]]
            correct = pd.DataFrame(correct_data, index=correct_index,
                columns=['Label','TrainCount'])
            if label_type_ld == 'left_lung_9999':
                correct = correct[correct['Label'].str.contains('left_lung')]
            correct = correct.sort_values(by = ['TrainCount','Label'], ascending=False)
            #Compare dfs
            assert eqc.dfs_equal_by_type(correct, output, numeric_cols=['TrainCount'], object_cols=['Label'])
            print('Passed test_return_traincount_flat() for',label_type_ld)
    
    def test_return_traincount_flat_filt(self):
        for label_type_ld in ['location_disease_9999','left_lung_9999']:
            tcffk_path = os.path.join('testing_delthis_DEID','9999_tcffk_num_num_DEID.csv')
            pairs = organize_labels.return_pairs(label_type_ld)
            output = organize_labels._return_traincount_flat_filt(tcffk_path, pairs)
            #Define correct answer
            global CORRECT_INDEX_FLAT_FILT, CORRECT_DATA_FLAT_FILT
            correct = pd.DataFrame(CORRECT_DATA_FLAT_FILT, index=CORRECT_INDEX_FLAT_FILT,
                columns=['Label','TrainCount'])
            if label_type_ld == 'left_lung_9999':
                correct = correct[correct['Label'].str.contains('left_lung')]
            correct = correct.sort_values(by = ['TrainCount','Label'], ascending=False)
            #Compare dfs
            assert eqc.dfs_equal_by_type(correct, output, numeric_cols=['TrainCount'], object_cols=['Label'])
            print('Passed test_return_traincount_flat_filt() for',label_type_ld)
    
    def test_return_traincount_flat_filt_keep(self):
        for label_type_ld in ['location_disease_9999','left_lung_9999']:
            tcffk_path = os.path.join('testing_delthis_DEID','9999_tcffk_num_num_DEID.csv')
            pairs = organize_labels.return_pairs(label_type_ld)
            #Define correct answer
            global CORRECT_INDEX_FLAT_FILT, CORRECT_DATA_FLAT_FILT
            correct = pd.DataFrame(CORRECT_DATA_FLAT_FILT, index=CORRECT_INDEX_FLAT_FILT,
                columns=['Label','TrainCount'])
            
            #Test for mincount_lung=0 and mincount_heart=0 (keep everything, even stuff
            #that has 0 count)
            output00 = organize_labels._return_traincount_flat_filt_keep(pairs, tcffk_path, mincount_lung=0, mincount_heart=0)
            correct00 = copy.deepcopy(correct)
            correct00['Decision'] = 'keep'
            if label_type_ld == 'left_lung_9999':
                correct00 = correct00[correct00['Label'].str.contains('left_lung')]
            correct00 = correct00.sort_values(by = ['TrainCount','Label'], ascending=False)
            assert eqc.dfs_equal_by_type(correct00, output00, numeric_cols=['TrainCount'], object_cols=['Label','Decision'])
            
            #Test for mincount_lung=1 and mincount_heart=2
            output12 = organize_labels._return_traincount_flat_filt_keep(pairs, tcffk_path, mincount_lung=1, mincount_heart=2)
            correct12 = copy.deepcopy(correct)
            if label_type_ld == 'location_disease_9999':
                decision12 = ['keep','exclude','keep','exclude','exclude','exclude',
                              'keep','exclude','keep','exclude','exclude','exclude',
                              'keep','keep','keep','keep','keep']
            elif label_type_ld == 'left_lung_9999':
                correct12 = correct12[correct12['Label'].str.contains('left_lung')]
                #Note that in the code, when we are loading left lung labels we actually
                #first get the location_disease traincount_flat_filt_keep df (i.e. we
                #use label_type_ld 'location_disease' initially), because we
                #want to exclude any left lung diseases that don't have a sufficient
                #count in the right lung. But for code testing purposes I'm going to
                #test as if label_type_ld is actually left_lung, just to make sure the code
                #works. The decision is different because the decision here is made only
                #based on the left lung counts and doesn't consider the right lung
                #counts.
                decision12 = ['keep','exclude','keep','keep','keep','exclude']
            correct12['Decision'] = decision12
            correct12 = correct12.sort_values(by = ['TrainCount','Label'], ascending=False)
            assert eqc.dfs_equal_by_type(correct12, output12, numeric_cols=['TrainCount'], object_cols=['Label','Decision'])
            
            #Test for mincount_lung=2 and mincount_heart=3
            output23 = organize_labels._return_traincount_flat_filt_keep(pairs, tcffk_path, mincount_lung=2, mincount_heart=3)
            correct23 = copy.deepcopy(correct)
            if label_type_ld == 'location_disease_9999':
                decision23 = ['exclude','exclude','exclude','exclude','exclude','exclude',
                              'exclude','exclude','exclude','exclude','exclude','exclude',
                              'exclude','keep','exclude','exclude','exclude']
            elif label_type_ld == 'left_lung_9999':
                correct23 = correct23[correct23['Label'].str.contains('left_lung')]
                decision23 = ['keep','exclude','exclude','keep','exclude','exclude']
            correct23['Decision'] = decision23
            correct23 = correct23.sort_values(by = ['TrainCount','Label'], ascending=False)
            assert eqc.dfs_equal_by_type(correct23, output23, numeric_cols=['TrainCount'], object_cols=['Label','Decision'])
            print('Passed test_return_traincount_flat_filt_keep() for',label_type_ld)
    
    def test_return_label_name_list(self):
        for label_type_ld in ['location_disease_9999','left_lung_9999']:
            binlabels_path = os.path.join('testing_delthis_DEID','imgtrain_BinaryLabels_DEID.pkl')
            pairs = organize_labels.return_pairs(label_type_ld)
            if label_type_ld == 'location_disease_9999':
                output = organize_labels._return_label_name_list(pairs, binlabels_path, label_type_ld, mincount_lung=1, mincount_heart=2)
                correct = ['h_mediastinum_cancer','h_mediastinum_mass',
                           'h_mediastinum_nodule','heart_cardiomegaly','heart_heart_failure',
                           'left_lung_atelectasis','left_lung_groundglass','right_lung_atelectasis',
                           'right_lung_groundglass']
            elif label_type_ld == 'left_lung_9999':
                output = organize_labels._return_label_name_list(pairs, binlabels_path, label_type_ld, mincount_lung=1, mincount_heart=2)
                correct = ['left_lung_atelectasis','left_lung_groundglass','left_lung_mass','left_lung_nodule']
            assert output == correct
            print('Passed test_return_label_name_list() for',label_type_ld)
    
    def test_make_custom_labels_df_location_disease(self):
        label_type_ld = 'location_disease_9999'
        binlabels_path = os.path.join('testing_delthis_DEID','imgtrain_BinaryLabels_DEID.pkl')
        pairs = organize_labels.return_pairs(label_type_ld)
        output = organize_labels.make_custom_labels_df(pairs, binlabels_path, label_type_ld, mincount_lung=1, mincount_heart=2)
        correct_index = ['patient_123','patient_456','patient_789']
        correct_columns = ['h_mediastinum_cancer','h_mediastinum_mass',
                           'h_mediastinum_nodule','heart_cardiomegaly',
                           'heart_heart_failure','left_lung_atelectasis',
                           'left_lung_groundglass','right_lung_atelectasis',
                           'right_lung_groundglass']
        correct_data = [[1,1,0,1,0,0,1,0,1],
                        [0,1,1,0,1,1,0,1,0],
                        [1,1,1,1,1,0,1,0,0]]
        correct = pd.DataFrame(correct_data,index=correct_index,columns=correct_columns)
        assert eqc.dfs_equal(output,correct)
        print('Passed test_make_custom_labels_df_location_disease()')
    
    def test_make_custom_labels_df_left_lung(self):
        label_type_ld = 'left_lung_9999'
        binlabels_path = os.path.join('testing_delthis_DEID','imgtrain_BinaryLabels_DEID.pkl')
        pairs = organize_labels.return_pairs(label_type_ld)
        output = organize_labels.make_custom_labels_df(pairs, binlabels_path, label_type_ld, mincount_lung=1, mincount_heart=2)
        correct_index = ['patient_123','patient_456','patient_789']
        correct_columns = ['left_lung_atelectasis','left_lung_groundglass',
                           'left_lung_mass','left_lung_nodule']
        correct_data = [[0,1,0,1],
                        [1,0,1,0],
                        [0,1,0,1]]
        correct = pd.DataFrame(correct_data,index=correct_index,columns=correct_columns)
        assert eqc.dfs_equal(output,correct)
        print('Passed test_make_custom_labels_df_left_lung()')
    
    def test_make_lung_labels_generic(self):
        #For more information see 2020-09-09-Updating-and-Testing-the-GenLung-Labels.docx
        label_meanings_fake = ['heart_cardiomegaly','heart_coronary_artery_disease',
                               'left_lung_atelectasis','left_lung_groundglass','left_lung_mass','left_lung_nodule','left_lung_pneumonia',
                               'right_lung_atelectasis','right_lung_groundglass','right_lung_mass','right_lung_nodule','right_lung_pneumonia']
        gr_truth_fake = torch.Tensor([0,1,1,0,0,0,0,0,1,1,0,0])
        sample_fake = {'gr_truth':gr_truth_fake,'volume_acc':'RHAA12345_6.npz'}
        note_acc_fake = 'AA12345'
        disease_generic_label_df_fake = pd.DataFrame([[0,0,0,0,1,1,1],
                                                      [0,1,1,0,1,0,1],
                                                      [1,1,1,0,0,0,1]],
            index = ['AA22222','AA12345','AA33333'],
            columns = ['cardiomegaly','coronary_artery_disease','atelectasis','groundglass','mass','nodule','pneumonia'])
        #Calculate output
        output_sample = custom_datasets.make_lung_labels_generic(sample_fake, label_meanings_fake, disease_generic_label_df_fake, note_acc_fake)
        
        #Check if output is correct
        assert (output_sample['gr_truth'] == torch.Tensor([0,1,1,1,1,0,1])).all()
        assert (output_sample['heart_gr_truth'] == torch.Tensor([0,1])).all()
        assert (output_sample['left_lung_gr_truth'] == torch.Tensor([1,0,0,0,1])).all()
        assert (output_sample['right_lung_gr_truth'] == torch.Tensor([1,1,1,0,1])).all()
        print('Passed test_make_lung_labels_generic()')

if __name__ == '__main__':
    unittest.main()
    