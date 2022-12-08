#identify_cts_to_plot.py

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

import numpy as np
import pandas as pd

def return_true_pos_with_highest_pred_prob(grtruth_path, predprob_path, mapping_path):
    """Return a pandas DataFrame with the following columns:
    ['VolumeAcc','VolumeAcc_ForOutput','Abnormality'] where 'VolumeAcc' is a
    real volume accession, 'VolumeAcc_ForOutput' is a fake volume accession
    that will be part of output file names, and 'Abnormality' is a
    comma-separated list of abnormalities to plot for that scan. 
    
    CT scans are selected for inclusion in the dictionaries as follows: for each
    abnormality label, select all the scans in which that abnormality is
    present (ground truth of 1) and then pick the scan with the
    highest predicted probability. In other words, identify the 'most positive
    true positives' in the data set.
    
    <grtruth_path> is the path to a ground truth file with columns that are
        abnormalities and an index of volume accessions.
        e.g. '/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/pred_probs/valid_grtruth_ep4.csv'
    <predprob_path> is the path to the corresponding predicted probabilities
        file, with columns that are abnormalities and an index of volume
        accessions.
        e.g. '/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/pred_probs/valid_predprob_ep4.csv'
    <mapping_path> is the path to the file that contains the mapping between
        fake and real accessions.
        e.g. /home/rlb61/data/img-hiermodel2/data/RADChestCT_Mapping/Conversion_PHI_to_DEID.csv
        Columns: ['MRN','NoteAcc','VolumeAcc','Set_Assigned','Subset_Assigned',
            'MRN_DEID','NoteAcc_DEID','VolumeAcc_DEID']"""
    grtruth = pd.read_csv(grtruth_path,header=0,index_col=0)
    predprob = pd.read_csv(predprob_path,header=0,index_col=0)
    mapping = pd.read_csv(mapping_path,header=0)
    
    outdf = pd.DataFrame(columns=['VolumeAcc','VolumeAcc_ForOutput','Abnormality'])
    curridx = 0
    
    #Fill in dicts
    print('VolumeAcc_DEID,','Abnormality,','VolAccPredProb,','AbnMean,','Factor') #header for printed output
    for abnormality in grtruth.columns.values.tolist():
        #First determine the real volume accession number:
        list_of_pos_scans = grtruth[grtruth[abnormality]==1].index.values.tolist()
        preds_of_pos_scans = predprob.filter(items=list_of_pos_scans,axis=0)[abnormality]
        #real_volacc stores the volume accession of the scan with the highest
        #predicted probability out of all scans positive for this abnormality
        real_volacc = preds_of_pos_scans.index.values.tolist()[preds_of_pos_scans.argmax()]
        real_volacc_predprob = preds_of_pos_scans[real_volacc]
        assert real_volacc_predprob==preds_of_pos_scans.max()
        
        #Now determine the fake volume accession corresponding to the real
        #volume accession
        fake_volacc_list = mapping[mapping['VolumeAcc']==real_volacc]['VolumeAcc_DEID'].values.tolist()
        assert len(fake_volacc_list)==1
        fake_volacc = fake_volacc_list[0]
        
        #Fill in the output df
        if real_volacc in outdf['VolumeAcc'].values.tolist():
            #then update the list of abnormalities for this volume acc
            chosenidx = outdf[outdf['VolumeAcc']==real_volacc].index.values.tolist()[0]
            outdf.at[chosenidx,'Abnormality'] = outdf.at[chosenidx,'Abnormality']+','+abnormality
        else:
            #then create a new entry for this volume acc
            outdf.at[curridx,'VolumeAcc'] = real_volacc
            outdf.at[curridx,'VolumeAcc_ForOutput'] = fake_volacc
            outdf.at[curridx,'Abnormality'] = abnormality
            curridx+=1
        
        #Print out
        abn_mean = np.mean(predprob[[abnormality]].values)
        print(fake_volacc+','+abnormality+','+str(round(real_volacc_predprob,5))+','
              +str(round(abn_mean,5))+','+str(round(real_volacc_predprob/abn_mean,2)))
    
    return outdf

    
