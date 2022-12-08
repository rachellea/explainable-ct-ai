#organize_pfts.py
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
import collections
import numpy as np
import pandas as pd

#File 1_ILD_Patient_Cohort.csv contains a mapping from the MRN that we provided
#to the updated MRN (sometimes a patient is duplicated and when the 2 records
#are merged that patient gets a new MRN). For now, ignore patients whose MRN
#changed because they might mess up the data split. For now, just use
#the MRN column found in the 3_ILD_Labs files and don't worry about salvaging
#the patients whose MRNs were changed. Deal with that later if time.

#Raw PFTs are in 3_ILD_Labs_2014_2016_with_units.csv and 3_ILD_Labs_2017_2020_with_units.csv
#columns: ['PAT_MRN_ID', 'PAT_ID', 'PRJ_LAB_RESULT_ID', 'PROJECT_COHORT_ID',
#'PROJECT_COHORT_DATE', 'ORDER_PROC_ID', 'PROC_ID', 'PROC_CODE', 'ORDER_DTTM',
#'RESULT_DTTM', 'COMPONENT_ID', 'COMPONENT_NAME', 'RESULT_VALUE',
#'RESULT_VALUE_NUM', 'REFERENCE_UNIT', 'PROC_NAME']

#Column descriptions (from notes on 2020-10-05)
#'PAT_MRN_ID' is the MRN
#'PRJ_LAB_RESULT_ID' is an electronic identifier for that instance of the
#    particular lab result (like a particular FEV1)
#'ORDER_PROC_ID' is unique identifier for the particular PFT overall.
#    ORDER_PROC_ID 123456 will be associated with all the values measured
#    at a particular 6 minute walk test/particular set of PFTs
#'PROJECT_COHORT_DATE': the date of the encounter associated with the
#    6 minute walk test.
#'ORDER_DTTM': the date time of the order.
#    Usually date ranges are filtered based off of ORDER_DTTM.
#RESULT_DTTM: this is when the test resulted. We can use RESULT_DTTM as the
#    best date time column, rather than ORDER_DTTM.
#'COMPONENT_ID', 'COMPONENT_NAME': this is like an identifier for "FEV1" conceptually (NOT a particular instance of FEV1)
#    all PFTs do NOT necessarily have the same set of COMPONENT_NAMEs within them
#    There are something like 112 different components
#    There are about 20 or so that are really common components
#    Then there are some that are much rarer components
#'CONTACT_DATE': same as 'PROJECT_COHORT_DATE'
#'PROC_ID' is the procedure ID for the 6-minute walk test
#'PROC_NAME' is the string description of the 6-minute walk test

def prepare_clean_pfts():
    results_dir = '/home/rlb61/data/img-hiermodel2/results/results_pfts/2021-01-13-Exploring-PFTs'
    all_pfts = load_all_pfts()
    obtain_common_components(all_pfts, results_dir)
    check_result_values(all_pfts)
    
####################
# Helper Functions #------------------------------------------------------------
####################
def load_all_pfts():
    raw_pfts_2014_to_2016 = pd.read_csv('/home/rlb61/data/img-hiermodel2/data/PFT_Data/3_ILD_Labs_2014_2016_with_units.csv',header=0)
    raw_pfts_2017_to_2020 = pd.read_csv('/home/rlb61/data/img-hiermodel2/data/PFT_Data/3_ILD_Labs_2017_2020_with_units.csv',header=0)
    assert raw_pfts_2014_to_2016.columns.values.tolist()==raw_pfts_2017_to_2020.columns.values.tolist()

    all_pfts = pd.concat([raw_pfts_2014_to_2016,raw_pfts_2017_to_2020],axis='index',ignore_index=True)
    assert all_pfts.shape[0]==(raw_pfts_2014_to_2016.shape[0]+raw_pfts_2017_to_2020.shape[0])
    return all_pfts

def obtain_common_components(all_pfts, results_dir):
    """Save a table Component_Summary_Stats.csv that summarizes the different
    components (kinds of lab tests), their units, their counts, and their
    mean/std values."""
    #Replace nan units with the string 'NO_UNITS'. This is so later on when we
    #do filtering using the reference units being 'equal to something' we
    #don't run into issues (since nan != nan in pandas)
    all_pfts['REFERENCE_UNIT'] = all_pfts['REFERENCE_UNIT'].fillna(value='NO_UNITS')
    
    #Check COMPONENT_ID and COMPONENT_NAME columns. Make sure the mapping
    #between COMPONENT_ID and COMPONENT_NAME seems reasonable (a particular
    #COMPONENT_ID should correspond to a particular COMPONENT_NAME)
    check_component_df = all_pfts[['COMPONENT_ID','COMPONENT_NAME']].drop_duplicates(ignore_index=True).sort_values(by='COMPONENT_ID')
    assert check_component_df.shape[0]==len(set(check_component_df['COMPONENT_ID']))
    assert check_component_df.shape[0]==len(set(check_component_df['COMPONENT_NAME']))
    
    #Init a dataframe to store the counts of components and their units
    component_stats_df = all_pfts[['COMPONENT_NAME','COMPONENT_ID','REFERENCE_UNIT']].drop_duplicates(ignore_index=True)
    
    #Rank the components from most commonly mentioned to least commonly
    #mentioned.
    #This component_rank_df will be used to fill in component_stats_df
    component_rank_df = rank_components(all_pfts)
    
    #Calculate the count of each component-units pair, where the pair is joined
    #into a string with a # separator
    #This component_unit_counts will be used to fill in component_stats_df
    component_unit_counts = collections.Counter(['#'.join([str(x) for x in pairlist]) for pairlist in all_pfts[['COMPONENT_NAME','REFERENCE_UNIT']].values.tolist()])
    
    #Fill in the dataframe with:
    #TOTAL_COUNT: from component_unit_counts, the total count of this
    #component with this particular kind of unit.
    #RANK: from component_stats_df, the rank of the component by its frequency
    #MEAN_RESULT_VALUE_NUM: mean result value for this component/unit excluding nans
    #STD_RESULT_VALUE_NUM: std result value for this component/unit excluding nans
    #NANS_EXCLUDED_COUNT: count of nans excluded from the mean and stdevs    
    for idx in component_stats_df.index.values.tolist():
        comp_name = component_stats_df.at[idx,'COMPONENT_NAME']
        ref_units = component_stats_df.at[idx,'REFERENCE_UNIT']
        
        #Store the rank, making use of the component_rank_df
        component_stats_df.at[idx,'RANK'] = component_rank_df.at[comp_name,'RANK']
        
        #Store the count, making use of the collections.Counter from before
        component_stats_df.at[idx,'TOTAL_COUNT'] = component_unit_counts[str(comp_name)+'#'+str(ref_units)]
        
        #Select out the result values for this component name and units combination
        relevant = all_pfts[(all_pfts['COMPONENT_NAME']==comp_name) & (all_pfts['REFERENCE_UNIT']==ref_units)]['RESULT_VALUE_NUM']
        
        #Figure out how many nans there are in the result values. Record the
        #total nan count, and then remove the nans from the result values
        #before the calculation of mean and stdev
        nancount = np.sum(np.isnan(relevant.values))
        component_stats_df.at[idx,'NANS_EXCLUDED_COUNT'] = nancount
        relevant_nonans = relevant.dropna(axis='index',how='any')
        assert relevant_nonans.shape[0]==(relevant.shape[0]-nancount)
        
        #Calculate the mean and standard deviation of the result value
        component_stats_df.at[idx,'MEAN_RESULT_VALUE_NUM'] = round(np.mean(relevant_nonans.values),4)
        component_stats_df.at[idx,'STD_RESULT_VALUE_NUM'] = round(np.std(relevant_nonans.values),4)
        
    component_stats_df = component_stats_df.sort_values(by=['RANK','TOTAL_COUNT'],ascending=[True,False])
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(component_stats_df)
    assert np.sum(component_stats_df['TOTAL_COUNT'].values)==all_pfts.shape[0]
    component_stats_df.to_csv(os.path.join(results_dir,'Component_Summary_Stats.csv'),header=True, index=False)

def rank_components(all_pfts):
    """Produce a dataframe that has the COMPONENT_NAME as the index, and two
    columns, COUNT which has the total occurrences of that component in
    <all_pfts>, and RANK which ranks the component with 1 being the most common
    to 104 being the least common."""
    component_counts = collections.Counter(all_pfts['COMPONENT_NAME'].values.tolist())
    component_rank_df = pd.DataFrame(index=list(component_counts.keys()),columns=['COUNT','RANK'])
    for component_name in component_rank_df.index.values.tolist():
        component_rank_df.at[component_name,'COUNT'] = component_counts[component_name]
    component_rank_df = component_rank_df.sort_values(by='COUNT',ascending=False)
    for numeric_idx in range(0,component_rank_df.shape[0]):
        #column 1 is the 'RANK' column
        component_rank_df.iat[numeric_idx,1] = numeric_idx+1
    return component_rank_df

def check_result_values(all_pfts):
    """Double check the RESULT_VALUE and RESULT_VALUE_NUM columns"""
    #Select the results values columns, and replace the non-numeric
    #string '*****' with np.nan so it will get dropped
    result_val_df = all_pfts[['RESULT_VALUE','RESULT_VALUE_NUM']].replace(to_replace='*****', value=np.nan).dropna(axis='index',how='any')
    
    #Check if the columns are numerically equal (as they should be)
    assert (np.isclose(result_val_df['RESULT_VALUE'].astype('float').values,
                       result_val_df['RESULT_VALUE_NUM'].astype('float').values)).all()
    
if __name__=='__main__':
    prepare_clean_pfts()
