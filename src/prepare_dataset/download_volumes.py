#download_volumes.py
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

import io
import os
import time
import pickle
import requests
import pydicom
import warnings
import datetime
import numpy as np
import pandas as pd
from requests_toolbelt.multipart import decoder

warnings.filterwarnings( 'ignore', '.*Unverified HTTPS.*' )

class DownloadData(object):
    def __init__(self, current_index, save_path):
        """
        <save_path>: path to save downloaded CTs e.g. /storage/rlb61-ct_images/vna/rlb61 """
        self.current_index = current_index
        print('Starting from current_index',current_index)
        self.r = get_token(verbose=True)
        self.save_path = save_path
        self.error_count = 0
        self.initialize_logging_df() #init logging dfs and counters
        self.read_in_accessions() #Read in tables of accession numbers to download
        
        #Download data
        #TODO: ones that were "failed" in earlier pulls
        self.read_and_save_everything_from_df(self.accs_30300_to_end)

    # Initialization #----------------------------------------------------------
    def initialize_logging_df(self):
        self.logging_df = pd.DataFrame(np.empty((1,7),dtype='str'),
                        columns = ['CurrentIndex','AccessionNumber',
                                   'SeriesNumber','ImageType','SeriesDescription',
                                   'NumSlices','ChosenOne'])
        self.logging_df_name = os.path.join('/home/rlb61/data/data_download_logging/',datetime.datetime.today().strftime('%Y-%m-%d')+'_Logging_Dataframe.csv')
        self.logging_df_idx = 0
        
    def read_in_accessions(self):
        #Read in the list of accession numbers you care about
        #First, report on the missing rate by reading in all lines of file, 
        #and then comparing to result of reading in while ignoring comments
        temp = pd.read_csv('/home/rlb61/data/2019-Rachel-RAI-Rubin-Project/INPUTFILES/MDM-abbrev_ct_chest_wo_contrast_w_3d_mips_protocol-20190701-1735.txt',
                              sep='|',header=None,index_col=False)
        print('All MDMcsv lines:',temp.shape)
        
        myfiles = pd.read_csv('/home/rlb61/data/2019-Rachel-RAI-Rubin-Project/INPUTFILES/MDM-abbrev_ct_chest_wo_contrast_w_3d_mips_protocol-20190701-1735.txt',
                              sep='|',header=None,index_col=False,comment='#',
                             names = ['MRN','AccessionNumber','Name','1','2','Date','3','Modality','Protocol','Study_UID','4','5'])
        print('\nMissing Percent:',1-myfiles.shape[0]/temp.shape[0])
        print('Only Present Lines:',myfiles.shape)
        
        #Skip the first 1300 as these were already downloaded
        use_files = myfiles.loc[1300:,:]
        print('Remove the first 1300:',use_files.shape)
        print('First indices:',use_files.index.values.tolist()[0:5])
        print('Last indices:',use_files.index.values.tolist()[-5:])
        
        self.accs_1300_to_6300 = use_files.loc[1300:6300,:]
        self.accs_6300_to_12300 = use_files.loc[6300:12300,:]
        self.accs_12300_to_18300 = use_files.loc[12300:18300,:]
        self.accs_18300_to_24300 = use_files.loc[18300:24300,:]
        self.accs_24300_to_30300 = use_files.loc[24300:30300,:]
        self.accs_30300_to_end = use_files.loc[30300:,:]

    # Read and Save scans key function #----------------------------------------
    def read_and_save_everything_from_df(self, accs_df):
        try:
            self.read_and_save(accs_df.loc[self.current_index:,:])
        except Exception as e:
            self.error_count+=1
            print(e)
            print('Error: waiting five minutes before restarting')
            time.sleep(300) #wait five minutes
            self.read_and_save(accs_df.loc[self.current_index:,:])
    
    # Read and Save helper functions #------------------------------------------
    def read_and_save(self, accs_df):
        total = accs_df.shape[0]
        for indexnum in accs_df.index.values.tolist():
            acc = accs_df.at[indexnum, 'AccessionNumber']
            self.r = get_token(verbose=False)
            t0 = time.time()
            self.get_one_series_custom(accession=acc, frames_thr=30)
            t1 = time.time()
            print('Finished ',acc,', index=',indexnum,' time=',round(t1-t0,2),'sec\n')
            self.current_index = indexnum
            self.error_count = max(0,self.error_count-1)
            
    def get_one_series_custom(self, accession, frames_thr):
        """Save all series for the specified accession number <accession>"""
        access_token = self.r['access_token']
        try:
            series_info = query_series_info(access_token, accession)
        except Exception as e:
            self.error_count+=1
            if self.error_count > 10:
                assert False, 'Quitting - too many errors'
            print(e,'\nfailed query_series_info on',accession)
            return(e)
        
        if series_info == []:
            self.error_count+=1
            if self.error_count > 10:
                assert False, 'Quitting - too many errors'
            print('failed - series is []',accession)
            return('fail')
        else:
            try:
                series_df = pd.DataFrame(series_info)
                #Keep only series with 'instances_number' greater than <frames_thr>:
                series_df = series_df[series_df['instances_number']>frames_thr]
                
                #get all possible scans for this accession:
                all_series_data = []
                for index, row in series_df.iterrows():
                    series_data = download_series(access_token, row['study_uid'], row['series_uid'] )
                    if series_data:
                        all_series_data.append(series_data)
                        
                #choose one scan to save:
                chosen_one, chosen_one_series_number, temp_logging_df = self.choose_scan_to_save(all_series_data)
                self.update_logging_df(accession, temp_logging_df)
                
                fname = os.path.join(self.save_path, row['accession']+'_'+str(chosen_one_series_number)+'.pkl' )
                of = open(fname, 'wb')
                pickle.dump(chosen_one, of)
                of.close()
                return('success')
                
            except Exception as e:
                self.error_count+=1
                if self.error_count > 10:
                    assert False, 'Quitting - too many errors'
                print(e,'\nfailed on',accession)
                return(e)

    def choose_scan_to_save(self, all_series_data):
        """Return the scan that should be saved.
        Variables:
        <all_series_data> is a list. Each element of the list is a sublist composed of
            pydicom.dataset.FileDataset objects in which each pydicom object is
            a DICOM for a single slice of a scan."""
        temp_logging_df = pd.DataFrame(np.empty((len(all_series_data),6),dtype='object'),
                        index = [x for x in range(len(all_series_data))],
                        columns = ['Data','SeriesNumber','SeriesDescription',
                                   'NumSlices','ImageType','ChosenOne'])
        
        for idx in range(len(all_series_data)):
            #data is itself a list of pydicom datasets
            data = all_series_data[idx]
            temp_logging_df.at[idx,'Data'] = data
            temp_logging_df.at[idx,'SeriesNumber'] = data[0]['0020','0011'].value #Series Number
            temp_logging_df.at[idx,'SeriesDescription'] = data[0]['008','103e'].value #Series Description
            temp_logging_df.at[idx,'NumSlices'] = len(data) #total number of slices
            #image type
            image_type = data[0]['008','008'].value #Image Type
            if 'ORIGINAL' in image_type:
                temp_logging_df.at[idx,'ImageType'] = 'ORIGINAL'
            elif ('DERIVED' in image_type) or ('SECONDARY' in image_type) or ('REFORMATTED' in image_type):
                temp_logging_df.at[idx,'ImageType'] = 'DERIVED'
            else:
                temp_logging_df.at[idx,'ImageType'] = '-'.join(list(image_type))
            temp_logging_df.at[idx,'ChosenOne'] = 'No'
        
        #Make selection
        keepers = temp_logging_df[temp_logging_df['ImageType']=='ORIGINAL'] #only keep ORIGINAL scans
        if keepers.shape[0]==0:
            return None, temp_logging_df
        #if we reach this point, there's at least one original scan
        #choose the original scan with the highest number of slices:
        max_slices = 0
        index_of_max_slices = None
        for keepidx in keepers.index.values.tolist():
            this_numslices = keepers.at[keepidx,'NumSlices']
            if this_numslices > max_slices:
                max_slices = this_numslices
                index_of_max_slices = keepidx
        
        #check if there was ever a situation where a non-original scan had
        #a greater number of slices than the original one you chose
        for tempidx in temp_logging_df.index.values.tolist():
            if temp_logging_df.at[tempidx,'NumSlices'] > max_slices:
                print('Non-Original Slice Exceeded Original Slices:',str(temp_logging_df.loc[tempidx,:]))
        
        #return chosen_one and temp_logging_df
        chosen_one = keepers.at[index_of_max_slices,'Data']
        chosen_one_series_number = keepers.at[index_of_max_slices,'SeriesNumber']
        temp_logging_df.at[index_of_max_slices,'ChosenOne'] = 'Yes'
        return chosen_one, chosen_one_series_number, temp_logging_df
                
    def update_logging_df(self, accession, temp_logging_df):
        """Update the overall logging df based on <temp_logging_df>
        and save the current version"""
        #Columns of self.logging_df are 'CurrentIndex','AccessionNumber',
        #'SeriesNumber','ImageType','SeriesDescription','NumSlices','ChosenOne'
        #Columns of temp_logging_df are 'Data','SeriesNumber','SeriesDescription',
        #'NumSlices','ImageType','ChosenOne'
        for idx in temp_logging_df.index.values.tolist():
            self.logging_df.at[self.logging_df_idx, 'CurrentIndex'] = self.current_index
            self.logging_df.at[self.logging_df_idx, 'AccessionNumber'] = accession
            self.logging_df.at[self.logging_df_idx, 'SeriesNumber'] = temp_logging_df.at[idx,'SeriesNumber']
            self.logging_df.at[self.logging_df_idx, 'ImageType'] = temp_logging_df.at[idx,'ImageType']
            self.logging_df.at[self.logging_df_idx, 'SeriesDescription'] = temp_logging_df.at[idx,'SeriesDescription']
            self.logging_df.at[self.logging_df_idx, 'NumSlices'] = temp_logging_df.at[idx,'NumSlices']
            self.logging_df.at[self.logging_df_idx, 'ChosenOne'] = temp_logging_df.at[idx,'ChosenOne']
            self.logging_df_idx+=1
        self.logging_df.to_csv(self.logging_df_name)
        
####################################
# Vendor Neutral Archive Functions #--------------------------------------------
####################################
def get_token(verbose=False):
    endpoint = 'https://oauth.oit.duke.edu/oidc/token'
    client_id = 'dia_apis_user'
    refresh_token = 'eyJhbGciOiJub25lIn0.eyJleHAiOjE1ODYzOTMyNzQsImp0aSI6ImQ0ZjljZDBiLTRlYjUtNGUwOS04YzBjLTlhNzc4M2NhNzEwMiJ9.'    
    if verbose: print(client_id,'\n',refresh_token)
    r = requests.post( endpoint, data = { 'client_id':client_id, 'grant_type':'refresh_token', 'refresh_token':refresh_token }, verify=False )
    if verbose: print( 'request ok:', r.ok )
    
    return r.json();

def query_series_info(access_token, accession_number): #formerly query_series_ct()
    """Pull information on the series specified by <accession_number>"""
    qendpoint = 'https://health-apis.duke.edu/qido-rs/dicom_web/series'
    qhead = { 'Authorization':'token {}'.format( access_token ),'Accept':'application/json' }
    g = requests.get( qendpoint, params={'AccessionNumber':accession_number, 'Modality':'CT', 'offset':0 }, headers=qhead, verify=False )
    
    # parse
    series = []
    if ( g.ok and ( g.status_code == 200 ) ):
        len( g.json() )
        series_raw = g.json()
        
        for item in g.json():
            #Here are the possible keys you can get:
            #dict_keys(['00080060', '0008103E', '00081190', '0020000E',
            #'00200011', '00201209', '00080020', '00080030', '00080050',
            #'00080061', '00080090', '00100010', '00100020', '00100030',
            #'00100040', '0020000D', '00200010', '00201206', '00201208'])
            #TODO: ask if you can get 008008 (Image Type) added to this list
            #of keys so that I can query instead of downloading the whole
            #DICOM just to check the Image Type
            series_ = {}
            series_['study_uid'] = item.get('0020000D')['Value'][0]
            series_['series_uid'] = item.get('0020000E')['Value'][0]
            series_['accession'] = item.get('00080050')['Value'][0]
            series_['instances_number'] = item.get('00201209')['Value'][0]
            series_['series_description'] = item.get('0008103E')['Value'][0]
            series.append(series_)
    return series

def download_series(access_token, study_uid, series_uid): #formerly get_series()
    wendpoint = 'https://health-apis.duke.edu/wado-rs/dicom_web/studies/{study_instance_uid}/series/{series_instance_uid}'.format(  study_instance_uid=study_uid, series_instance_uid=series_uid )
    whead = {'Authorization': 'token {}'.format( access_token ),'Accept':'multipart/related; type=application/dicom'}
    h = requests.get( wendpoint, headers=whead, verify=False )
    ds = [];
    if ( h.ok and ( h.status_code == 200 ) ):
        mp_data = decoder.MultipartDecoder.from_response( h )
        ds = [ pydicom.dcmread( io.BytesIO( part.content ) ) for part in mp_data.parts ]
    return ds

if __name__ == '__main__':
    DownloadData(current_index = 35888)