#eval_by_scan_attr.py
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
import datetime
import collections
import numpy as np
import pandas as pd
import sklearn.metrics

#import sys
#sys.path.insert(0, '/home/rlb61/data/img-hiermodel2/')
#from load_dataset import custom_datasets

from src.evals import nominal
from src.evals import delong_ci

def run_all(descriptor, grtruth_file, predprob_file, phi_or_deid):
    calculate_corr_between_scan_attr_and_abns(descriptor, grtruth_file)
    
    calculate_model_perf_by_scan_attr(descriptor, grtruth_file,
        predprob_file, phi_or_deid)
    
def calculate_corr_between_scan_attr_and_abns(descriptor, grtruth_file):
    """Produce a three-column dataframe in which the first two columns
    contain strings of scan attributes or abnormalities, and the final
    Correlation column includes the quantified correlation between
    the scan attributes and abnormalities."""
    results_dir = return_results_dir(descriptor)
    
    #TODO: later on, calculate the correlation based on the whole data set
    #instead of just a subset. (Tricky part: genericizing lung labels happens
    #one example at a time in custom_datasets.py)
    grtruth = pd.read_csv(grtruth_file, header=0, index_col=0)
    
    metadata_df, volacc_col, cat_attrs = read_in_scan_metadata_df('phi', grtruth)
    
    #Merge in the ground truth, so that when we calculate correlation, we
    #can include correlations between the ground truth labels and the scan
    #attributes
    merged = pd.merge(grtruth,metadata_df,left_index=True,right_on='full_filename_npz')
    merged = merged.set_index(keys=volacc_col)
    
    #Calculate correlations between all of the columns. This will involve
    #correlations between different abnormalities, between different
    #scan attributes, and between abnormalities and scan attributes.
    #It will also involve comparing categorical w/ categorical variables,
    #continuous w/ continuous variables, and categorical w/ continuous
    #variables.
    #corrs includes ['corr'] and ['ax']
    #where corrs['corr'] is a dataframe that has all abnormalities/attributes
    #as columns AND as the index (i.e. the index and columns are the same)
    #and the values are the correlations
    corrs = nominal.associations(dataset=merged,
                 nominal_columns=cat_attrs)
    
    #Pick out dataframe of correlations
    corrdf = corrs['corr']
    
    #Check symmetry
    assert np.allclose(corrdf.values, corrdf.values.T)
    
    #Check that the correlation of a thing with itself is always 1
    for abnattr in corrdf.columns.values.tolist():
        if corrdf.at[abnattr,abnattr]!=1.0:
            print('TODO Resolve Corr Error:',abnattr,'=',corrdf.at[abnattr,abnattr],'instead of 1.0!')
    
    #Determine all possible unique pairs (excluding identical pairs [a,a])
    pairs=[]
    for abnattr_a in corrdf.columns.values.tolist():
        for abnattr_b in corrdf.columns.values.tolist():
            if abnattr_a != abnattr_b: #exclude identical pairs
                pair = sorted([abnattr_a,abnattr_b])
                if pair not in pairs:
                    pairs.append(pair)
    
    #Rearrange correlation df into a 2-column format sorted by strength 
    twocol = pd.DataFrame(pairs,columns=['A','B'])
    twocol['Correlation']=0.0
    for idx in twocol.index.values.tolist():
        twocol.at[idx,'Correlation'] = corrdf.at[twocol.at[idx,'A'],twocol.at[idx,'B']]
    twocol = twocol.sort_values(by='Correlation',ascending=False)
    twocol.to_csv(os.path.join(results_dir,'correlation_dataframe.csv'),header=True,index=False)
    
    #Filter by StationName
    station_name = twocol[twocol['A'].str.contains('StationName') | twocol['B'].str.contains('StationName')]
    station_name.to_csv(os.path.join(results_dir,'correlation_dataframe_station_name.csv'),header=True,index=False)

    #TODO deal with the plot
    #TODO: when making the plot, filter according to what's actually correlated
    #TODO: make the plot bigger

def calculate_model_perf_by_scan_attr(descriptor, grtruth_file, predprob_file, phi_or_deid):
    """
    Variables
    <grtruth_file>: path to a CSV for a pandas dataframe. index is the
        accession number. Columns are the labels. Values are 1 or 0.
    <predprob_file>: path to a CSV for a pandas dataframe. index is the
        accession number. Columns are the labels. Values are probabilities.
    <phi_or_deid>: either 'phi' or 'deid' to indicate whether the accession
        numbers are PHI (i.e. the original accession numbers) or deidentified
        (i.e. the fake deidentified accession numbers)
    """
    results_dir = return_results_dir(descriptor)
    
    assert phi_or_deid in ['phi','deid']
    
    #Read in the ground truth and predicted probabilities
    grtruth = pd.read_csv(grtruth_file, header=0, index_col=0)
    predprob = pd.read_csv(predprob_file, header=0, index_col=0)
    
    #Read in the file that defines the scan attributes for each scan
    #and also obtain the attributes we want to analyze
    metadata_df, volacc_col, cat_attrs = read_in_scan_metadata_df(phi_or_deid, grtruth)
    
    #Calculate AUROC and average precision subdivided according to different
    #attr_value_counts of each attribute
    calculate_perf_differences_by_scan_attr(results_dir, grtruth, predprob,
                                            metadata_df, volacc_col)

####################
# Helper Functions #------------------------------------------------------------
####################
def return_results_dir(descriptor):
    results_dir = os.path.join('/home/rlb61/data/img-hiermodel2/results/results_2021/',datetime.datetime.today().strftime('%Y-%m-%d')+'_'+descriptor)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    return results_dir

def read_in_scan_metadata_df(phi_or_deid, grtruth):
    #Read in the metadata with the scan attributes
    if phi_or_deid == 'phi':
        metadata_df = pd.read_csv('/home/rlb61/data/img-hiermodel2/data/RADChestCT_PHI/Final_CT_Scan_Preproc_and_All_Metadata_Summary.csv',header=0)
        volacc_col = 'full_filename_npz'
    elif phi_or_deid == 'deid':
        metadata_df = pd.read_csv('/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/CT_Scan_Metadata_Complete_DEID.csv',header=0)
        volacc_col = 'VolumeAcc_DEID'
    
    #Add the PatientAgeYears column based on the PatientAge column
    metadata_df = add_patient_age_in_years(metadata_df)
    
    #Infer the subsets_list before you take away the Subset_Assigned column
    subsets_list = return_inferred_subsets_list(metadata_df, grtruth, volacc_col)
    
    #Filter by subsets_list. This reduces the number of rows in the dataframe
    #(the number of examples) so that the only examples remaining are those
    #that have ground truth.
    metadata_df = metadata_df[metadata_df['Subset_Assigned'].isin(subsets_list)]
    
    #Define numeric scan attributes of interest
    #Note we are excluding SpacingBetweenSlices because it is missing about
    #half the time, and isn't a super important attribute to look at anyway:
    #(Pdb) Counter(metadata_df['SpacingBetweenSlices'].values.tolist())
    #Counter({'5.000000': 19618, 'ATTR_VALUE_NOT_FOUND': 15630, '-0.625000': 896, '2.500000': 116, '1.250000': 46, '0.625000': 7, '0.500000': 3})
    num_attrs = ['SliceThickness','PatientAgeYears',
        'orig_square','orig_numslices','orig_slope','orig_inter',
        'orig_yxspacing','orig_zdiff']
    
    #Define categorical scan attributes of interest
    cat_attrs = ['Manufacturer','ManufacturerModelName','InstitutionName',
        'StationName','SoftwareVersions','ConvolutionKernel',
        'PatientSex','EthnicGroup','IterativeReconAnnotation',
        'IterativeReconConfiguration','IterativeReconLevel',
        'ProtocolName','ReconAlgo','ReconAlgoManuf']
    
    #Keep only the relevant columns
    keep_cols = [volacc_col]+num_attrs+cat_attrs
    metadata_df = metadata_df[keep_cols]
    
    #Ensure that numeric columns don't contain any non-numeric values
    #and that they don't contain any missing values
    for colname in num_attrs:
        assert metadata_df.dtypes[colname]==np.float64
        assert metadata_df[colname].isna().sum()==0
    
    return metadata_df, volacc_col, cat_attrs

def add_patient_age_in_years(metadata_df):
    """Add a column called PatientAgeYears which is based on the PatientAge
    column"""
    #Convert PatientAge format into years and save as PatientAgeYears column
    metadata_df['PatientAgeYears'] = -1.0
    for idx in metadata_df.index.values.tolist():
        age = metadata_df.at[idx,'PatientAge']
        if isinstance(age,str) and len(age)>0:
            if 'Y' in age:
                #'067Y' --> 67
                num = age.replace('Y','').lstrip('0')
                if len(num)>0: #i.e. anything other than '000Y' (which believe it or not is an age)
                    age_years = int(num)
                    metadata_df.at[idx,'PatientAgeYears'] = age_years
            elif 'M' in age:
                num = age.replace('M','').lstrip('0')
                if len(num)>0:
                    age_months = float(num) #e.g. '022M' --> 22
                    age_years = age_months/12.0
                    metadata_df.at[idx,'PatientAgeYears'] = age_years
    return metadata_df

def calculate_perf_differences_by_scan_attr(results_dir, grtruth, predprob, metadata_df, volacc_col):
    attribute_columns = metadata_df.columns.values.tolist()
    attribute_columns.remove(volacc_col)
    for colname in attribute_columns:
        auroc_df, auroc_ci_df = calculate_perf_diff_for_one_attr(colname, grtruth, predprob, metadata_df, volacc_col)
        auroc_df.to_csv(os.path.join(results_dir, colname+'_AUROC.csv'),header=True,index=True)
        auroc_ci_df.to_csv(os.path.join(results_dir, colname+'_AUROC_Confidence_Intervals.csv'),header=True,index=True)

def calculate_attr_value_counts(metadata_df, colname):
    """Return a dictionary where the keys are the different values that
    a particular scan attribute <colname> can take on, and the values are
    integer counts indicating how often the scan attribute takes on that value.
    Rare options (count <30) are grouped together."""
    #raw_attr_value_counts has keys that are different values that this
    #particular scan attribute takes on. The associated values are counts.
    #Example for colname = 'StationName':
    #Counter({'CC_CT1': 230, 'DMP_CT2': 210, 'CC_CT4': 194, 'ct01': 185,
    #         'CC_CT2': 171, 'CC_CT5': 167, 'CTC1': 116, 'CTC3': 111,
    #         'CTAWP73718': 102, 'DMPRAD3FORCE': 98, 'CCCT3Revo': 75,
    #         'DMP_CT1': 61, 'ipct1': 59, 'DRAH_CT2': 58, 'ctb5': 52,
    #         'J3GECT750': 29, 'CC_CT3': 26, 'ect2': 22, 'IPCT1': 20,
    #         'ect1': 18, 'ctj3': 17, 'CaryCT1': 14, 'ctj1': 14, 'EDCT1': 12,
    #         'MPCT1': 9, 'DRHCT1': 6, 'DRHCT2': 6, 'hrct': 2, 'sct1': 1})
    raw_attr_value_counts = collections.Counter(metadata_df[colname].values.tolist())
    
    #Group together rare values
    rare_value = ''
    rare_value_count = 0
    final_attr_value_counts = {}
    for key in list(raw_attr_value_counts.keys()):
        if raw_attr_value_counts[key] < 30:
            rare_value = '#'.join([str(rare_value), str(key)])
            rare_value_count += raw_attr_value_counts[key]
        else:
            final_attr_value_counts[key] = raw_attr_value_counts[key]
    final_attr_value_counts[rare_value.strip('#')] = rare_value_count
    
    #Example final_attr_value_counts for colname = 'StationName':
    #{'DMPRAD3FORCE': 98, 'CTC1': 116, 'CC_CT1': 230, 'CCCT3Revo': 75,
    # 'ct01': 185, 'DMP_CT2': 210, 'CTC3': 111, 'CC_CT5': 167, 'CC_CT4': 194,
    # 'DMP_CT1': 61, 'DRAH_CT2': 58, 'ipct1': 59, 'CTAWP73718': 102,
    # 'CC_CT2': 171, 'ctb5': 52,
    # 'IPCT1&J3GECT750&CaryCT1&DRHCT1&EDCT1&ect2&DRHCT2&ctj1&MPCT1&ect1&ctj3&CC_CT3&sct1&hrct': 196}
    return final_attr_value_counts

def obtain_attribute_values_list(attribute_value):
    """Inspect the string <attribute_value> to determine if it is an
    hashtag-delimited string of attribute values created in
    the function calculate_attr_value_counts().
    If it's an hashtag-delimited string, then split it up again into its
    original components and return it.
    If it's just one concept put it in a list by itself and return it."""
    #The attribute_value might be an aggregation of rare attribute_values,
    #delimited by an hashtag, e.g. a#b#c. We need to parse out the
    #individual attribute values
    if isinstance(attribute_value,str) and ('#' in attribute_value):
        attribute_values_list = attribute_value.split('#')
        if attribute_values_list[0].replace('.','',1).isdigit():
            if '.' in attribute_value: #floats
                attribute_values_list = [float(x) for x in attribute_values_list]
           # else: #integers (e.g. '20&50&40&60')
           #     attribute_values_list = [int(x) for x in attribute_values_list]
    else:
        attribute_values_list = [attribute_value]
    return attribute_values_list

def calculate_perf_diff_for_one_attr(colname, grtruth, predprob, metadata_df, volacc_col):
    attr_value_counts = calculate_attr_value_counts(metadata_df, colname)
    
    #TODO - calculate p values w/ FDR or Bonferroni correction!
    #So you can find significant differences
    #TODO - write function that will identify the most statistically
    #significant differences across all the analyses.
    #TODO - add a column that calculates the max-min per row.
    
    #auroc_df has Count and abnormality labels as the index, and
    #different values that this particular scan attribute takes on as the
    #columns.
    auroc_df = pd.DataFrame(index=['Count']+grtruth.columns.values.tolist(),columns=list(attr_value_counts.keys()))
    auroc_ci_df = pd.DataFrame(index=['Count']+grtruth.columns.values.tolist(),columns=list(attr_value_counts.keys())) #for CIs
    
    for attribute_value in list(attr_value_counts.keys()): #e.g. attribute_value is 'DMPRAD3FORCE' if colname='StationName'
        auroc_df.at['Count',attribute_value] = attr_value_counts[attribute_value]
        auroc_ci_df.at['Count',attribute_value] = attr_value_counts[attribute_value]
        
        attribute_values_list = obtain_attribute_values_list(attribute_value)
        
        #Select the relevant accession numbers, i.e. the accessions for the
        #scans that have this particular scan characteristic
        relevant_accs = metadata_df[metadata_df[colname].isin(attribute_values_list)][volacc_col].values.tolist()
        assert len(relevant_accs)==attr_value_counts[attribute_value]
        
        #ground truth and pred probs for the relevant accs only
        #these dfs have the relevant accs as the index, and all abnormality
        #labels as the columns
        relevant_grtruth = grtruth.filter(items=relevant_accs,axis='index')
        relevant_predprob = predprob.filter(items=relevant_accs,axis='index')
        assert relevant_grtruth.index.values.tolist()==relevant_predprob.index.values.tolist()
                
        #Calculate AUROC
        for abnormality_label in relevant_grtruth.columns.values.tolist():
            y_true = relevant_grtruth[abnormality_label]
            y_score = relevant_predprob[abnormality_label]
            
            #there must be both 1s and 0s in y_true to calculate AUROC:
            if ((1 in y_true.values.tolist()) and (0 in y_true.values.tolist())):
                #Calculate AUROC value with sklearn
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score, pos_label = 1)
                auc = sklearn.metrics.auc(fpr, tpr)
                
                #Calculate AUROC value and confidence interval with delong_ci.py:
                #output format of calc_auc_ci() is (0.7143757881462799, array([0.68191345, 0.74683813]))
                auc_with_ci = delong_ci.calc_auc_ci(y_true, y_score)
                
                #Sanity check - ensure the AUC values calculated in different ways
                #are basically equal
                assert abs(auc_with_ci[0]-auc)<10e-9
                
                #Save
                auroc_df.at[abnormality_label, attribute_value] = round(auc_with_ci[0],4)
                auroc_ci_df.at[abnormality_label, attribute_value] = str([round(x,4) for x in auc_with_ci[1].tolist()])
            
    return auroc_df, auroc_ci_df

def return_inferred_subsets_list(metadata_df, grtruth, volacc_col):
    example_volacc = grtruth.index.values.tolist()[0]
    example_setname = metadata_df[metadata_df[volacc_col]==example_volacc]['Subset_Assigned'].values.tolist()[0] #e.g. 'imgvalid_a'
    if 'valid' in example_setname:
        return ['imgvalid_a']
    elif 'test' in example_setname:
        return ['imgtest_a','imgtest_b','imgtest_c','imgtest_d']

    