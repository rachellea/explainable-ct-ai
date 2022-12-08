#main.py
import os
from attn_analysis import multipanel_group_plot

if __name__=='__main__':
    #get ids
    raw_filenames = os.listdir('/storage/rlb61-data/img-hiermodel2/results/2020-11-14_ValidAttnAnalysisTARGETED_of_2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart/attn_2dplot_gradcam-vanilla')
    ids = list(set(['_'.join(z.split('_')[0:2]) for z in raw_filenames])) #note these are actually real IDs
    
    multipanel_group_plot.run(results_dir='/storage/rlb61-data/img-hiermodel2/results/2020-11-14-TARGETED-MultiPanel-Group-Plots',
                              mask_dir='/storage/rlb61-data/img-hiermodel2/results/2020-11-14_ValidAttnAnalysisTARGETED_of_2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart',
                              base_dir='/storage/rlb61-data/img-hiermodel2/results/2020-11-14_ValidAttnAnalysisTARGETED_of_2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart',
                              ids_list=ids,
                              ids_type='real')
