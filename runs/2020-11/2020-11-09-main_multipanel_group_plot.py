#main.py

from attn_analysis import multipanel_group_plot

if __name__=='__main__':
    fakeids_list = ['111','222','333','444','555',
                    '666','777','888','999','123',
                    '234','345','456','567','678',
                    '789','chAAA','chBBB','chCCC',
                    'chDDD','chEEE','chFFF','chGGG',
                    'chHHH','chIII','chJJJ','chLLL']
    multipanel_group_plot.run(results_dir='/storage/rlb61-data/img-hiermodel2/results/2020-11-09-MultiPanel-Group-Plots',
                              mask_dir='/storage/rlb61-data/img-hiermodel2/results/2020-11-09_ValidAttnAnalysis_of_2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart',
                              base_dir='/storage/rlb61-data/img-hiermodel2/results/2020-11-09_ValidAttnAnalysis_of_2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart',
                              ids_list=fakeids_list,
                              ids_type='fake')