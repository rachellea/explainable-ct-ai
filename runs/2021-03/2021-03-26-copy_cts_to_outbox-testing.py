#2021-03-26-copy_cts_to_outbox-testing.py

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
import pandas as pd

from src.deid import make_movie

def chunks(lst, n):
    #https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__=='__main__':
    #copy the 2000 train / 1000 val subset to the Honest Broker outbox
    #minus the pediatric scans
    #columns: MRN_DEID,NoteAcc_DEID,Set_Assigned,Subset_Assigned
    #index: an integer you can ignore
    train_part = pd.read_csv('/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/predefined_subsets/2020-01-10-imgtrain_random2000_DEID.csv',header=0)['NoteAcc_DEID'].values.tolist()[0:800]
    valid_part = pd.read_csv('/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/predefined_subsets/2020-01-10-imgvalid_a_random1000_DEID.csv',header=0)['NoteAcc_DEID'].values.tolist()[0:400]
    full_list = train_part+valid_part
    full_list.sort()
    
    #Remove peds scans
    nopeds = pd.read_csv('/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/CT_Scan_Metadata_Complete_DEID_nopeds.csv',header=0)
    assert nopeds.shape[0]==35747
        
    final_full_list = [x for x in full_list if x in nopeds['NoteAcc_DEID'].values.tolist()]
    
    print('After removal of pediatric scans, total is',len(final_full_list))
    print('Scans to be removed:')
    for f in final_full_list:
        print(f)
    
    #Add .npz suffix to the note accession to form the volume accession
    final_full_list = [x+'.npz' for x in final_full_list]
    
    #Copy all the volumes in to the Honest Broker outbox
    
    #DELTHIS:
    final_full_list = final_full_list[0:10]
    
    #Make visualizations
    for sublist in chunks(final_full_list,10):
        #Determine path to save output GIF
        sublist_first_filename = sublist[0] #e.g. AA123456.npz
        sublist_last_filename = sublist[-1] #e.g. AA123506.npz
        movie_dir = '/home/rlb61/data/img-hiermodel2/results/results_2021/2021-03-26-GIF-Testing'
        savepath = os.path.join(movie_dir,sublist_first_filename.replace('.npz','')+'_to_'+sublist_last_filename.replace('.npz',''))
        
        make_movie.make_whole_gif(ctvol_directory='/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-DEID/',
                                  ctvol_filenames_list=sublist,
                                  savepath=savepath)
    