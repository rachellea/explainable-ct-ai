#report_summary_of_metadata_recon.py
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

import numpy as np
import pandas as pd

#metadata file:
#columns Accession,MRN,Set_Assigned,Set_Should_Be,Subset_Assigned,status,
#status_reason,full_filename_pkl,SliceThickness,SpacingBetweenSlices,Manufacturer,
#ManufacturerModelName,StudyDescription,StudyDate,PatientAge,Modality,
#InstitutionName,StationName,SoftwareVersions,ConvolutionKernel,
#PatientSex,EthnicGroup
metadata_misc = pd.read_csv('/home/rlb61/data/PreprocessVolumes/2020-05-14_metadata_extraction/CT_Scan_MetaData_Extraction_Log_File.csv',header=0)
#drop status and status_reason because we want to keep only the final
#download status (i.e. whether the volume downloaded successfully or not,
#rather than whether all of these particular metadata fields were accessed
#or not)
metadata_misc = metadata_misc.drop(columns=['status','status_reason'])
print('rows in metadata_misc',metadata_misc.shape[0])

#metadata file reconstruction algorithm:
#Accession,MRN,Set_Assigned,Set_Should_Be,Subset_Assigned,status,
#status_reason,full_filename_pkl,IterativeReconAnnotation,
#IterativeReconConfiguration,IterativeReconLevel,ProtocolName
metadata_recon = pd.read_csv('/home/rlb61/data/PreprocessVolumes/2020-05-23_metadata_recon_extraction/CT_Scan_MetaData_Extraction_Recon_Log_File.csv',header=0)
metadata_recon = metadata_recon.drop(columns=['status','status_reason'])
print('rows in raw metadata_recon',metadata_recon.shape[0])

#Preprocessing log file which contains metadata too
#36316 volumes in the dataset
#columns you need are Accession and status
#all columns:['Accession', 'MRN', 'Set_Assigned', 'Set_Should_Be',
# 'Subset_Assigned', 'status', 'status_reason', 'full_filename_pkl',
# 'full_filename_npz', 'orig_square', 'orig_numslices', 'orig_slope',
# 'orig_inter', 'orig_yxspacing', 'orig_zdiff_set', 'orig_zdiff',
# 'zdiffs_all_equal', 'orig_orientation', 'orig_gantry_tilt',
# 'final_square', 'final_numslices', 'final_spacing']
metadata_preproc = pd.read_csv('/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/CT_Scan_Preprocessing_Log_File_FINAL_SMALL.csv',header=0)
print('rows in raw metadata_preproc',metadata_recon.shape[0])

#Merge
shared = ['Accession','MRN','Set_Assigned','Set_Should_Be','Subset_Assigned','full_filename_pkl']
metadata_temp = pd.merge(metadata_misc, metadata_recon,  how='inner', left_on=shared, right_on=shared, validate='one_to_one')
metadata_all = pd.merge(metadata_temp, metadata_preproc, how='inner', left_on=shared, right_on=shared, validate='one_to_one')
print('metadata all merged raw:',metadata_all.shape[0])

#filter metadata_df based on available accs:
metadata_all = metadata_all[metadata_all['status']=='success']
assert metadata_all.shape[0]==36316
print('metadata all merged final:',metadata_all.shape[0])

#Obtain the reconstruction algorithm:
#From Justin Solomon:
#For Siemens scanners, you can use the ConvolutionKernel field. If that field
#contains two values then that implies that the series was reconstructed with
#an iterative algorithm.
#For example:
#    ConvolutionKernel == B31f implies FBP,
#    ConvolutionKernel == "['Br40d', '2']" implies iterative
# 
# For GE scanners, there are some private DICOM fields that will only exist
#if the series was reconstructed with an iterative algorithm:
# 0x0053,0x1040
# 0x0053,0x1042
# 0x0053,0x1043
# So you can just check if any of them exist to classify those series.

metadata_all['ReconAlgo']=''
metadata_all['ReconAlgoManuf']=''
for idx in metadata_all.index.values.tolist():
    if metadata_all.at[idx,'Manufacturer'] == 'SIEMENS':
        convkernel = metadata_all.at[idx,'ConvolutionKernel']
        if (('[' in convkernel) and (',' in convkernel) and (']' in convkernel)):
            metadata_all.at[idx,'ReconAlgo'] = 'Iterative'
            metadata_all.at[idx,'ReconAlgoManuf'] = 'Iterative_SIEMENS'
        else:
            metadata_all.at[idx,'ReconAlgo'] = 'FBP'
            metadata_all.at[idx,'ReconAlgoManuf'] = 'FBP_SIEMENS'
        
    elif metadata_all.at[idx,'Manufacturer'] == 'GE MEDICAL SYSTEMS':
        if metadata_all.at[idx,'IterativeReconAnnotation'] == 'ATTR_VALUE_NOT_FOUND':
            metadata_all.at[idx,'ReconAlgo'] = 'FBP'
            metadata_all.at[idx,'ReconAlgoManuf'] = 'FBP_GEMEDICAL'
        else:
            metadata_all.at[idx,'ReconAlgo'] = 'Iterative'
            metadata_all.at[idx,'ReconAlgoManuf'] = 'Iterative_GEMEDICAL'
    
#Save the
metadata_all.to_csv('Final_CT_Scan_Preproc_and_All_Metadata_Summary.csv',header=True,index=False)

#Summary stats for recon algorithm
for attr in ['ReconAlgo','ReconAlgoManuf']:
    sel = metadata_all[attr]
    counts = sel.value_counts(normalize=False)
    freqs = sel.value_counts(normalize=True)
    summary = pd.DataFrame(np.zeros((counts.shape[0],2)),columns=['Count','Percent'],
                           index=counts.index.values.tolist())
    for idx in counts.index.values.tolist():
        summary.at[idx,'Count'] = counts[idx]
        summary.at[idx,'Percent'] = freqs[idx]
    totals = summary.sum(axis=0)
    summary.at['Total','Count'] = totals['Count']
    summary.at['Total','Percent'] = totals['Percent']
    summary['Count'] = summary['Count'].astype('int') #integer counts
    summary['Percent'] = [round(100*x,2) for x in summary['Percent']] #round percent to 2 decimal places
    summary.to_csv(attr+'Summary.csv',header=True,index=True)
    print('Saved summary df of',attr)
