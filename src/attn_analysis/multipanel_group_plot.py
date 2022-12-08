#multipanel_group_plot.py
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
import random

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ioff()
import matplotlib.image as mpimg

matplotlib.rc('font',**{'size':80})

LABEL_MEANINGS = ['h_great_vessel_aneurysm', 'h_great_vessel_atherosclerosis', 'h_great_vessel_calcification', 'h_great_vessel_catheter_or_port',
                  'h_great_vessel_dilation_or_ectasia', 'h_great_vessel_postsurgical', 'h_great_vessel_scattered_calc', 'h_mediastinum_calcification',
                  'h_mediastinum_cancer', 'h_mediastinum_lymphadenopathy', 'h_mediastinum_mass', 'h_mediastinum_nodule', 'h_mediastinum_opacity',
                  'heart_atherosclerosis', 'heart_cabg', 'heart_calcification', 'heart_cardiomegaly', 'heart_catheter_or_port', 'heart_coronary_artery_disease',
                  'heart_heart_failure', 'heart_heart_valve_replacement', 'heart_pacemaker_or_defib', 'heart_pericardial_effusion', 'heart_pericardial_thickening',
                  'heart_postsurgical', 'heart_scattered_calc', 'heart_stent', 'heart_sternotomy', 'heart_transplant', 'lung_air_trapping',
                  'lung_airspace_disease', 'lung_aspiration', 'lung_atelectasis', 'lung_bandlike_or_linear', 'lung_bronchial_wall_thickening',
                  'lung_bronchiectasis', 'lung_bronchiolectasis', 'lung_calcification', 'lung_cancer', 'lung_catheter_or_port', 'lung_cavitation',
                  'lung_chest_tube', 'lung_clip', 'lung_consolidation', 'lung_cyst', 'lung_density', 'lung_emphysema', 'lung_fibrosis', 'lung_granuloma',
                  'lung_groundglass', 'lung_honeycombing', 'lung_infection', 'lung_inflammation', 'lung_interstitial_lung_disease', 'lung_lesion',
                  'lung_lucency', 'lung_lung_resection', 'lung_lymphadenopathy', 'lung_mass', 'lung_mucous_plugging', 'lung_nodule', 'lung_nodulegr1cm',
                  'lung_opacity', 'lung_plaque', 'lung_pleural_effusion', 'lung_pleural_thickening', 'lung_pneumonia', 'lung_pneumothorax',
                  'lung_postsurgical', 'lung_pulmonary_edema', 'lung_reticulation', 'lung_scarring', 'lung_scattered_calc', 'lung_scattered_nod',
                  'lung_septal_thickening', 'lung_soft_tissue', 'lung_staple', 'lung_suture', 'lung_transplant', 'lung_tree_in_bud']

def make_group_plot(realid, fakeid, selected_label, results_dir, base_dir, mask_dir):
    """Make a multi-panel plot by loading PNGs that were created with the
    attention analysis pipeline and organizing them according to the
    model and the attention mechanism
    
    <base_dir> is a string with the path to the directory of the baseline model
        attn analysis results
        Example: '/storage/rlb61-data/img-hiermodel2/results/2020-11-10_ValidAttnAnalysis_of_2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart'
    <mask_dir> is a string with the path to the directory of the mask model
        attn analysis results
        Example: '/storage/rlb61-data/img-hiermodel2/results/2020-11-10_ValidAttnAnalysis_of_2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart'
    """
    hires_dir = 'attn_2dplot_hirescam'
    gradcam_dir = 'attn_2dplot_gradcam-vanilla'
    
    base_gradcam_rank1, subdir_bg1 = read_in_image(os.path.join(base_dir,gradcam_dir),realid+'_'+selected_label+'rank1_2dplot.png')
    base_gradcam_rank2, subdir_bg2 = read_in_image(os.path.join(base_dir,gradcam_dir),realid+'_'+selected_label+'rank2_2dplot.png')
    base_gradcam_rank3, subdir_bg3 = read_in_image(os.path.join(base_dir,gradcam_dir),realid+'_'+selected_label+'rank3_2dplot.png')
    base_gradcam_rank4, subdir_bg4 = read_in_image(os.path.join(base_dir,gradcam_dir),realid+'_'+selected_label+'rank4_2dplot.png')
    base_gradcam_rank5, subdir_bg5 = read_in_image(os.path.join(base_dir,gradcam_dir),realid+'_'+selected_label+'rank5_2dplot.png')
    assert subdir_bg1==subdir_bg2==subdir_bg3==subdir_bg4==subdir_bg5
    
    base_hirescam_rank1, subdir_bh1 = read_in_image(os.path.join(base_dir,hires_dir),realid+'_'+selected_label+'rank1_2dplot.png')
    base_hirescam_rank2, subdir_bh2 = read_in_image(os.path.join(base_dir,hires_dir),realid+'_'+selected_label+'rank2_2dplot.png')
    base_hirescam_rank3, subdir_bh3 = read_in_image(os.path.join(base_dir,hires_dir),realid+'_'+selected_label+'rank3_2dplot.png')
    base_hirescam_rank4, subdir_bh4 = read_in_image(os.path.join(base_dir,hires_dir),realid+'_'+selected_label+'rank4_2dplot.png')
    base_hirescam_rank5, subdir_bh5 = read_in_image(os.path.join(base_dir,hires_dir),realid+'_'+selected_label+'rank5_2dplot.png')
    assert subdir_bh1==subdir_bh2==subdir_bh3==subdir_bh4==subdir_bh5
    
    mask_gradcam_rank1, subdir_mg1 = read_in_image(os.path.join(mask_dir,gradcam_dir),realid+'_'+selected_label+'rank1_2dplot.png')
    mask_gradcam_rank2, subdir_mg2 = read_in_image(os.path.join(mask_dir,gradcam_dir),realid+'_'+selected_label+'rank2_2dplot.png')
    mask_gradcam_rank3, subdir_mg3 = read_in_image(os.path.join(mask_dir,gradcam_dir),realid+'_'+selected_label+'rank3_2dplot.png')
    mask_gradcam_rank4, subdir_mg4 = read_in_image(os.path.join(mask_dir,gradcam_dir),realid+'_'+selected_label+'rank4_2dplot.png')
    mask_gradcam_rank5, subdir_mg5 = read_in_image(os.path.join(mask_dir,gradcam_dir),realid+'_'+selected_label+'rank5_2dplot.png')
    assert subdir_mg1==subdir_mg2==subdir_mg3==subdir_mg4==subdir_mg5
    
    mask_hirescam_rank1, subdir_mh1 = read_in_image(os.path.join(mask_dir,hires_dir),realid+'_'+selected_label+'rank1_2dplot.png')
    mask_hirescam_rank2, subdir_mh2 = read_in_image(os.path.join(mask_dir,hires_dir),realid+'_'+selected_label+'rank2_2dplot.png')
    mask_hirescam_rank3, subdir_mh3 = read_in_image(os.path.join(mask_dir,hires_dir),realid+'_'+selected_label+'rank3_2dplot.png')
    mask_hirescam_rank4, subdir_mh4 = read_in_image(os.path.join(mask_dir,hires_dir),realid+'_'+selected_label+'rank4_2dplot.png')
    mask_hirescam_rank5, subdir_mh5 = read_in_image(os.path.join(mask_dir,hires_dir),realid+'_'+selected_label+'rank5_2dplot.png')
    assert subdir_mh1==subdir_mh2==subdir_mh3==subdir_mh4==subdir_mh5
    
    fig, ax = plt.subplots(nrows=5,ncols=4,figsize=(64,70))
    fig.subplots_adjust(hspace=0, wspace=0)
    
    ax[0,0].set_title('Base + GradCAM\n'+subdir_bg1)
    ax[0,0].imshow(base_gradcam_rank1)
    ax[1,0].imshow(base_gradcam_rank2)
    ax[2,0].imshow(base_gradcam_rank3)
    ax[3,0].imshow(base_gradcam_rank4)
    ax[4,0].imshow(base_gradcam_rank5)
    
    ax[0,1].set_title('Base + HiResCAM\n'+subdir_bh1)
    ax[0,1].imshow(base_hirescam_rank1)
    ax[1,1].imshow(base_hirescam_rank2)
    ax[2,1].imshow(base_hirescam_rank3)
    ax[3,1].imshow(base_hirescam_rank4)
    ax[4,1].imshow(base_hirescam_rank5)
    
    ax[0,2].set_title('Mask + GradCAM\n'+subdir_mg1)
    ax[0,2].imshow(mask_gradcam_rank1)
    ax[1,2].imshow(mask_gradcam_rank2)
    ax[2,2].imshow(mask_gradcam_rank3)
    ax[3,2].imshow(mask_gradcam_rank4)
    ax[4,2].imshow(mask_gradcam_rank5)
    
    ax[0,3].set_title('Mask + HiResCAM\n'+subdir_mh1)
    ax[0,3].imshow(mask_hirescam_rank1)
    ax[1,3].imshow(mask_hirescam_rank2)
    ax[2,3].imshow(mask_hirescam_rank3)
    ax[3,3].imshow(mask_hirescam_rank4)
    ax[4,3].imshow(mask_hirescam_rank5)
    
    ax[0,0].set_ylabel('rank 1')
    ax[1,0].set_ylabel('rank 2')
    ax[2,0].set_ylabel('rank 3')
    ax[3,0].set_ylabel('rank 4')
    ax[4,0].set_ylabel('rank 5')
    
    # selected_organ = ''
    # for organ in ['lung','heart','great_vessel','mediastinum']:
    #     if organ in selected_label:
    #         selected_organ = organ
    # 
    # abnormality = ' '.join(selected_label.replace(selected_organ,'').split('_'))
    # fig.suptitle(abnormality+' ('+selected_organ+')', fontsize=120)
    
    fig.suptitle(selected_label)
    
    #Remove all axis labels
    for row in [0,1,2,3,4]:
        for col in [1,2,3]:
            ax[row,col].spines['top'].set_visible(False)
            ax[row,col].spines['right'].set_visible(False)
            ax[row,col].spines['left'].set_visible(False)
            ax[row,col].spines['bottom'].set_visible(False)
            ax[row,col].tick_params(axis='both', left=False, top=False, 
                                    right=False, bottom=False, 
                                    labelleft=False, labeltop=False, 
                                    labelright=False, labelbottom=False)
    
    #Remove all axis labels except for the leftmost label of column 0
    for row in [0,1,2,3,4]:
        for col in [0]:
            ax[row,col].spines['top'].set_visible(False)
            ax[row,col].spines['right'].set_visible(False)
            ax[row,col].spines['left'].set_visible(False)
            ax[row,col].spines['bottom'].set_visible(False)
            ax[row,col].set_yticklabels([]) #get rid of the numbers
            ax[row,col].tick_params(axis='both', left=False, top=False, 
                                    right=False, bottom=False, 
                                    labelleft=True, labeltop=False, 
                                    labelright=False, labelbottom=False)
    
    fig.subplots_adjust(hspace=0)
    #fig.tight_layout()
    #fig.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(results_dir,fakeid+'_'+selected_label+'.png'))
    plt.close()


def read_in_image(specific_directory, image_filename):
    """Read in the image <image_filename> and infer which subdirectory
    of <specific_directory> it's in: 'g1p1', 'g1p0', 'g0p1', or 'g0p0.' """
    for subdir in ['g1p1', 'g1p0', 'g0p1', 'g0p0']:
        full_path = os.path.join(os.path.join(specific_directory, subdir), image_filename)
        if os.path.isfile(full_path):
            return mpimg.imread(full_path), subdir
    assert False, 'You should have found the file by now'

def run(results_dir, mask_dir, base_dir, ids_list, ids_type):
    """"Run group plot code on all examples
    <results_dir> is the directory for storing the multipanel group plots
    <ids_type> is either 'fake' or 'real'
    """
    global LABEL_MEANINGS
    assert len(LABEL_MEANINGS)==80
    
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
        
    assert ids_type in ['fake','real']
    if ids_type=='real':
        randint_id = random.randint(0, 10000)
        
    for realid in ids_list:
        print(realid)
        if ids_type=='real':
            fakeid = str(randint_id) #make up a fake ID that was started from a random integer
            print('FakeID for',realid,'is',fakeid) #record the mapping in the printed output in case you need it later
            randint_id+=1 #increment so that the next scan will have a different fake ID assigned
        else: #realid is already a fakeid
            fakeid = realid
        
        for selected_label in LABEL_MEANINGS:
            make_group_plot(realid, fakeid, selected_label, results_dir, base_dir, mask_dir)

def run_simple(results_dir, mask_dir, base_dir):
    """Run group plot code on true positives. Infer which scan/abnormality combos
    need to be visualized."""
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    def clean_filename(file):
        return file.replace('rank1_2dplot.png','').replace('rank2_2dplot.png','').replace('rank3_2dplot.png','').replace('rank4_2dplot.png','').replace('rank5_2dplot.png','')
    
    #e.g. 'val26876_lung_pulmonary_edemarank1_2dplot.png', 'val27186_lung_pleural_effusionrank4_2dplot.png', 'val25404_lung_cavitationrank4_2dplot.png',...
    #which then gets cleaned to 'val26876_lung_pulmonary_edema', 'val27186_lung_pleural_effusion', 'val25404_lung_cavitation',...
    files = list(set([clean_filename(x) for x in os.listdir(os.path.join(os.path.join(mask_dir,'attn_2dplot_hirescam'),'g1p1'))]))
    
    for file in files:
        print('Working on',file)
        realid = file.split('_')[0]
        selected_label = '_'.join(file.split('_')[1:])
        make_group_plot(realid, realid, selected_label, results_dir, base_dir, mask_dir)