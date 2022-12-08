#collect_volume_metadata.py
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
import timeit
import pickle
import pydicom #this might be used only implicitly, by unpickling pydicom objects
import datetime
import pandas as pd

class CollectMetadata(object):
    def __init__(self, old_log_file_path):
        """Variables:
        <old_log_file_path>:
            path to a log file from a previous run of this module. If this path
            is not the empty string '' then the logging dataframe will be
            initialized from this old log file and the module will pick up
            where it left off."""
        self.old_log_file_path = old_log_file_path
        
        #Set up input and output paths and necessary filenames
        self.set_up_paths()
        self.set_up_identifiers()
        self.set_up_logdf()
        
        #Run
        self.extract_metadata_from_all_pace_data() 
           
    #############
    # PACE Data #---------------------------------------------------------------
    #############
    def set_up_paths(self):
        """Set up the paths for input and output data"""
        #Location of the identifiers files which contain PACE and BD2K ids
        #for the train, valid, and test sets:
        self.identifiers_path = '/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/'
        
        #Location to save output of this script:
        self.logdir = os.path.join('/home/rlb61/data/PreprocessVolumes/',datetime.datetime.today().strftime('%Y-%m-%d')+'_metadata_extraction')
        if not os.path.isdir(self.logdir):
            os.mkdir(self.logdir)
        
        #Location of the "dirty" CT DICOMs from which to extract metadata
        self.dirty_path = '/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-DICOM'
            
    def set_up_identifiers(self):
        """Read in the identifiers file which defines the data splits as well
        as the fake filename translation. Also read in all of the available
        filenames."""
        #Read in identifiers file
        #identifiers file includes: ['MRN','Accession','Set_Assigned','Set_Should_Be','Subset_Assigned']
        ids_file = os.path.join(self.identifiers_path,'all_identifiers.csv')
        self.ids_df = pd.read_csv(ids_file, header=0, index_col = 'Accession')
        self.note_accessions = self.ids_df.index.values.tolist() #e.g. AA12345
        
        #The actual filenames of the downloaded volumes may be prefixed with
        #additional letters besides AA, will have an underscore followed by
        #a number at the end, and will have the .pkl file extension at the end.
        #For example RHAA12345_3.pkl. Thus they are not exact string matches
        #to the accessions in self.note_accessions.
        #Also until the data is completely downloaded, self.volume_accessions
        #will have fewer entries in it than self.note_accessions because
        #self.volume_accessions represents what's actually been downloaded
        #while self.note_accessions represents everything that COULD BE downloaded
        self.volume_accessions = os.listdir(self.dirty_path)
    
    def set_up_logdf(self):
        #dicom_attributes is a list of DICOM attributes to attempt to extract
        self.dicom_attributes = ['SliceThickness','SpacingBetweenSlices','Manufacturer',
                            'ManufacturerModelName','StudyDescription',
                            'StudyDate','PatientAge','Modality','InstitutionName',
                            'StationName','SoftwareVersions','ConvolutionKernel',
                            'PatientSex','EthnicGroup']
        if self.old_log_file_path == '':
            #there is no preexisting log file; initialize from scratch
            self.logdf = copy.deepcopy(self.ids_df)
            for colname in ['status','status_reason','full_filename_pkl']:
                self.logdf[colname]=''
            for attribute_name in self.dicom_attributes:
                if attribute_name not in self.logdf.columns.values.tolist():
                    self.logdf[attribute_name]=''
                    self.logdf[attribute_name]=self.logdf[attribute_name].astype('object')
        else:
            #there is a preexisting log file; start from this.
            self.logdf = pd.read_csv(self.old_log_file_path,header=0,index_col=0)
            self.update_note_accessions_from_existing()
        
    def update_note_accessions_from_existing(self):
        """If there is a preexisting log file defined by self.old_log_file_path,
        then update self.note_accessions so that it includes only the note_accs
        which need to be processed in this run."""
        print('Initializing logging df from previous version at ',self.old_log_file_path)
        print('Total available note accessions:',len(self.note_accessions))
        volume_not_found = self.logdf[self.logdf['status_reason']=='volume_not_found'].index.values.tolist()
        print('number of volume_not_found',len(volume_not_found))
        wrong_submatch = self.logdf[self.logdf['status_reason']=='processed_wrong_submatch_accession'].index.values.tolist()
        print('number of processed_wrong_submatch_accession',len(wrong_submatch))
        self.note_accessions = volume_not_found + wrong_submatch
        print('Total note accessions that will be processed:',len(self.note_accessions))
        
    def extract_metadata_from_all_pace_data(self):
        """Save a log file documenting the cleaning process and metadata."""
        t0 = timeit.default_timer()
        count = 0
        for note_acc in self.note_accessions:
            #note_acc is e.g. AA12345. full_filename_pkl is e.g. RHAA12345_3.pkl
            #full_filename_pkl is the full filename of the raw dirty CT.
            #full_filename_pkl will be 'fail' if not found.
            full_filename_pkl = self.find_full_filename_pkl(note_acc)
            self.logdf.at[note_acc,'full_filename_pkl'] = full_filename_pkl
            
            if full_filename_pkl == 'fail':
                self.logdf.at[note_acc,'status'] = 'fail'
                self.logdf.at[note_acc,'status_reason'] = 'volume_not_found'
                print('\tFailed on',note_acc,'because volume was not found')
            
            if full_filename_pkl != 'fail':
                self.extract_ctvol_metadata(full_filename_pkl, note_acc)
            count+=1
            if count % 20 == 0: self.report_progress_and_save_logfile(count,t0,note_acc)
    
    def report_progress_and_save_logfile(self,count,t0,note_acc):
        t1 = timeit.default_timer()
        percent = round(float(count)/len(self.note_accessions), 2)*100
        elapsed = round((t1 - t0)/60.0,2)
        print('Finished up to',note_acc,count,'=',percent,'percent. Elapsed time:',elapsed,'min')
        try:
            self.logdf.to_csv(os.path.join(self.logdir,'CT_Scan_MetaData_Extraction_Log_File.csv'))
            print('Saved log file')
        except Exception as e:
            print('Could not save log file this time due to Exception',e)
    
    def find_full_filename_pkl(self, note_acc):
        """e.g. if <note_acc>==AA12345 return RHAA12345_3.pkl, which is the
        full filename corresponding to this accession number.
        Previously I was checking if note_acc in full_filename_pkl but that does
        not work because if note_acc = AA1234, then it could match with
        multiple possible full_filename_pkls including AA123456 and AA12345678.
        Therefore I have to split the full_filename_pkl and ensure an exact
        match with the first part of the name."""
        for full_filename_pkl in self.volume_accessions:
            full_filename_pkl_extract = full_filename_pkl.split('_')[0].replace('RH','').replace('B','') #e.g. RHAA1234_6.pkl --> AA1234
            if note_acc == full_filename_pkl_extract:
                return full_filename_pkl
        return 'fail'
    
    def extract_ctvol_metadata(self, full_filename_pkl, note_acc):
        """Read in the pickled CT volume and extract metadata."""
        #Load volume. Format: python list. Each element of the list is a
        #pydicom.dataset.FileDataset that contains metadata as well as pixel
        #data. Each pydicom.dataset.FileDataset corresponds to one slice.
        raw = pickle.load(open(os.path.join(self.dirty_path, full_filename_pkl),'rb'))
        oneslice = raw[0]
        self.logdf.at[note_acc,'status'] = 'success'
        for attribute_name in self.dicom_attributes:
            try:
                self.logdf.at[note_acc,attribute_name] = oneslice.data_element(attribute_name).value
            except:
                self.logdf.at[note_acc,attribute_name] = 'ATTR_VALUE_NOT_FOUND'
                self.logdf.at[note_acc,'status'] = 'fail'
                self.logdf.at[note_acc,'status_reason'] = 'failed_on_'+attribute_name

###########
# Running #---------------------------------------------------------------------
###########
if __name__ == '__main__':
    CollectMetadata(old_log_file_path = '')
