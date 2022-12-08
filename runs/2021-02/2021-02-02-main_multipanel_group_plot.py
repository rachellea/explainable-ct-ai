#main.py

from src.attn_analysis import multipanel_group_plot

if __name__=='__main__':
    fakeids_list = ['chAAA', 'chBBB', 'chCCC', 'chDDD', 'chFFF', 'chGGG',
        'chIII', 'chJJJ', 'chLLL', 'ad0001', 'ad0002', 'ad0003', 'ad0004',
        'ad0005', 'ad0006', 'ad0007', 'ad0008', 'ad0009', 'ad00010', 'ad00011',
        'ad00012', 'ad00013', 'ad00014', 'ad00015', 'ad00016', 'ad00017',
        'ad00018', 'ad00019', '222', '333', '444', '666', '777', '888', '999',
        '123', '345', '456', '567', '678']
    
    multipanel_group_plot.run(results_dir='/home/rlb61/data/img-hiermodel2/results/results_2021/2021-02-02-MultiPanel-Group-Plots/',
                              mask_dir='/home/rlb61/data/img-hiermodel2/results/results_2021/2021-01-23_ValidAttnAnalysis_of_2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/',
                              base_dir='/home/rlb61/data/img-hiermodel2/results/results_2021/2021-01-23_ValidAttnAnalysis_of_2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart/',
                              ids_list=fakeids_list,
                              ids_type='fake')
    