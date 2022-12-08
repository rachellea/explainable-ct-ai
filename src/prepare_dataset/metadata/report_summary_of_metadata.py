#report_summary_of_metadata.py
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
import heapq
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg') #so that it does not attempt to display via SSH
import matplotlib.pyplot as plt
plt.ioff() #turn interactive plotting off
matplotlib.rcParams.update({'font.size': 18})

#metadata file:
#columns Accession,MRN,Set_Assigned,Set_Should_Be,Subset_Assigned,status,
#status_reason,full_filename_pkl,SliceThickness,SpacingBetweenSlices,Manufacturer,
#ManufacturerModelName,StudyDescription,StudyDate,PatientAge,Modality,
#InstitutionName,StationName,SoftwareVersions,ConvolutionKernel,
#PatientSex,EthnicGroup
metadata_df = pd.read_csv('/home/rlb61/data/PreprocessVolumes/2020-05-14_metadata_extraction/CT_Scan_MetaData_Extraction_Log_File.csv',header=0)
print('rows in raw metadata_df',metadata_df.shape[0])

#36316 volumes in the dataset
#columns you need are Accession and status
#all columns:['Accession', 'MRN', 'Set_Assigned', 'Set_Should_Be',
# 'Subset_Assigned', 'status', 'status_reason', 'full_filename_pkl',
# 'full_filename_npz', 'orig_square', 'orig_numslices', 'orig_slope',
# 'orig_inter', 'orig_yxspacing', 'orig_zdiff_set', 'orig_zdiff',
# 'zdiffs_all_equal', 'orig_orientation', 'orig_gantry_tilt',
# 'final_square', 'final_numslices', 'final_spacing']
available_accs_df = pd.read_csv('/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/CT_Scan_Preprocessing_Log_File_FINAL_SMALL.csv',header=0)
available_accs_df = available_accs_df[available_accs_df['status']=='success']
print('available accs',available_accs_df.shape[0])

#filter metadata_df based on available accs:
metadata_df = metadata_df[metadata_df['Accession'].isin(available_accs_df['Accession'].values.tolist())]
print('rows in filtered metadata df',metadata_df.shape[0])

#dict for summary of categorical DICOM attributes from CT_Scan_MetaData_Extraction_Log_File.csv
dicom_attributes1 = {'source':metadata_df,
                     'attr_cols': ['SliceThickness','SpacingBetweenSlices',
                                   'Manufacturer','ManufacturerModelName','StudyDescription',
                                   'Modality','InstitutionName', 'StationName',
                                   'SoftwareVersions','ConvolutionKernel','PatientSex','EthnicGroup']}

#dict for summary of categorical DICOM attributes from CT_Scan_Preprocessing_Log_File_FINAL_SMALL.csv
dicom_attributes2 = {'source':available_accs_df,
                     'attr_cols':['orig_square', 'orig_numslices', 'orig_slope',
                                  'orig_inter', 'orig_yxspacing', 'orig_zdiff_set', 'orig_zdiff',
                                  'zdiffs_all_equal', 'orig_orientation', 'orig_gantry_tilt',
                                  'final_square', 'final_numslices', 'final_spacing']}

for dictionary in [dicom_attributes1,dicom_attributes2]:
    for attr in dictionary['attr_cols']:
        df = dictionary['source']
        assert df.shape[0]==36316
        sel = df[attr]
        counts = sel.value_counts(normalize=False)
        freqs = sel.value_counts(normalize=True)
        summary = pd.DataFrame(np.zeros((counts.shape[0],2)),columns=['Count','Frequency'],
                               index=counts.index.values.tolist())
        for idx in counts.index.values.tolist():
            summary.at[idx,'Count'] = counts[idx]
            summary.at[idx,'Frequency'] = freqs[idx]
        totals = summary.sum(axis=0)
        summary.at['Total','Count'] = totals['Count']
        summary.at['Total','Frequency'] = totals['Frequency']
        summary['Count'] = summary['Count'].astype('int') #integer counts
        summary['Frequency'] = [round(x,3) for x in summary['Frequency']] #round freq to 3 decimal places
        summary.to_csv(os.path.join('SummaryTablesAndFigs',attr+'Summary.csv'),header=True,index=True)
        print('Saved summary df of',attr)

#Summarize StudyDate
dates = metadata_df['StudyDate']
print('Minimum date:',min(dates))
print('Median date:',np.median(dates))
print('Maximum date:',max(dates))

#Summarize PatientAge
ages_overall = []
for age in metadata_df['PatientAge']:
    if isinstance(age,str) and len(age)>0:
        if 'Y' in age:
            #'067Y' --> 67
            num = age.replace('Y','').lstrip('0')
            if len(num)>0: #i.e. anything other than '000Y' (which believe it or not is an age)
                age_years = int(num) 
                ages_overall.append(age_years)
        elif 'M' in age:
            num = age.replace('M','').lstrip('0')
            if len(num)>0:
                age_months = float(num) #e.g. '022M' --> 22
                age_years = age_months/12.0
                ages_overall.append(age_years)

print('Ages available for',len(ages_overall),'patients')
print('5 minimum ages:',heapq.nsmallest(5,ages_overall))
print('Median age:',np.median(ages_overall))
print('5 maximum ages:',heapq.nlargest(5,ages_overall))

#Plot ages histogram
#http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
plt.figure(figsize=(10, 7.5))  
ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
plt.xticks(fontsize=18)  
plt.yticks(fontsize=18)
plt.xlabel("Age (Years)", fontsize=20)  
plt.ylabel("Number of Patients", fontsize=20)  
plt.hist(ages_overall, bins=25, rwidth = 0.7, color="#3F5D7D")
plt.savefig(os.path.join('SummaryTablesAndFigs','AgesHistogram.pdf'), bbox_inches="tight")
plt.close()
