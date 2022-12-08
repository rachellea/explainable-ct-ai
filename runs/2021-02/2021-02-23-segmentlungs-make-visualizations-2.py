#2021-02-23-segmentlungs-make-visualizations-2.py
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
        [['trn19747.npz','trn19747',  'mrn08068','imgtrain_extra','imgtrain'],
        ['trn13222.npz','trn13222',  'mrn03085','imgtrain_extra','imgtrain'],
        ['trn19140.npz','trn19140',  'mrn00147','imgtrain_extra','imgtrain'],
        ['trn11755.npz','trn11755',  'mrn00701','imgtrain_extra','imgtrain'],
        ['trn11823.npz','trn11823',  'mrn09140','imgtrain_extra','imgtrain'],
        ['trn15982.npz','trn15982',  'mrn05871','imgtrain_extra','imgtrain'],
        ['trn17844.npz','trn17844',  'mrn00700','imgtrain_notetrain','imgtrain'],
        ['trn15234.npz','trn15234',  'mrn01996','imgtrain_extra','imgtrain'],
        ['trn15986.npz','trn15986',  'mrn09594','imgtrain_extra','imgtrain'],
        ['trn19754.npz','trn19754',  'mrn05507','imgtrain_extra','imgtrain'],
        ['trn19551.npz','trn19551',  'mrn06881','imgtrain_notetrain','imgtrain'],
        ['trn17893.npz','trn17893',  'mrn06835','imgtrain_extra','imgtrain'],
        ['trn19730.npz','trn19730',  'mrn11024','imgtrain_notetrain','imgtrain']],
        columns = ['VolumeAcc_DEID','NoteAcc_DEID','MRN_DEID','Set_Assigned','Subset_Assigned'])
    volume_log_df.to_csv(os.path.join(results_dir,'mini_volume_log_df.csv'),header=True,index=False)
    
    segmentlungs.SegmentLung(volume_log_df_path = os.path.join(results_dir,'mini_volume_log_df.csv'),
                 ct_scan_path = '/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-DEID/', #Load from remote drive for pace-henao-02 machine
                 results_dir = results_dir,
                 visualize = True)
    