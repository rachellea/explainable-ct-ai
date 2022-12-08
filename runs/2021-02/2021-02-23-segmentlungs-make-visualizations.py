#2021-02-23-segmentlungs-make-visualizations.py
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

from src.seg import segmentlungs

if __name__=='__main__':
    results_dir = '/home/rlb61/data/img-hiermodel2/results/results_2021/2021-02-23-SegmentLungs-Visualizations'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    #Use only a subset of scans
    volume_log_df = pd.DataFrame(
        [['trn19675.npz', 'trn19675',  'mrn09729',  'imgtrain_notetrain',  'imgtrain'],
        ['trn19678.npz', 'trn19678',  'mrn11866',  'imgtrain_extra',      'imgtrain'],
        ['trn19676.npz', 'trn19676',  'mrn05678',  'imgtrain_extra',      'imgtrain'],
        ['trn19677.npz', 'trn19677',  'mrn12547',  'imgtrain_extra',      'imgtrain'],
        ['trn11230.npz', 'trn11230',  'mrn01097',  'imgtrain_extra',      'imgtrain'],
        ['trn16593.npz', 'trn16593',  'mrn05358',  'imgtrain_extra',      'imgtrain']],
        columns = ['VolumeAcc_DEID','NoteAcc_DEID','MRN_DEID','Set_Assigned','Subset_Assigned'])
    volume_log_df.to_csv(os.path.join(results_dir,'mini_volume_log_df.csv'),header=True,index=False)
    
    segmentlungs.SegmentLung(volume_log_df_path = os.path.join(results_dir,'mini_volume_log_df.csv'),
                 ct_scan_path = '/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-DEID/', #Load from remote drive for pace-henao-02 machine
                 results_dir = results_dir,
                 visualize = True)
    