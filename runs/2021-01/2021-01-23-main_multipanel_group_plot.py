#main.py

from src.attn_analysis import multipanel_group_plot

if __name__=='__main__':
    fakeids_list = ['chEEE','chHHH','111','555','234','789']
    multipanel_group_plot.run(results_dir='/home/rlb61/data/img-hiermodel2/results/results_2021/2021-01-23-MultiPanel-Group-Plots/',
                              mask_dir='/home/rlb61/data/img-hiermodel2/results/results_2021/2021-01-23_ValidAttnAnalysis_of_2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/',
                              base_dir='/home/rlb61/data/img-hiermodel2/results/results_2021/2021-01-23_ValidAttnAnalysis_of_2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart/',
                              ids_list=fakeids_list,
                              ids_type='fake')
    
    
    
    
    
    
    
    
    
    
    
        
