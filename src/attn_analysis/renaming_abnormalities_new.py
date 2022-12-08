#renaming_abnormalities_new.py
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

#For making figures with human-readable captions

GV_AND_MEDIA_COLS = ['h_great_vessel_aneurysm', 'h_great_vessel_atherosclerosis',
              'h_great_vessel_calcification', 'h_great_vessel_catheter_or_port',
              'h_great_vessel_dilation_or_ectasia', 'h_great_vessel_postsurgical',
              'h_great_vessel_scattered_calc', 'h_mediastinum_calcification',
              'h_mediastinum_cancer', 'h_mediastinum_lymphadenopathy',
              'h_mediastinum_mass', 'h_mediastinum_nodule', 'h_mediastinum_opacity']

GV_AND_MEDIA_COLS_RENAME = {'h_great_vessel_aneurysm':'great vessel aneurysm',
                                  'h_great_vessel_atherosclerosis':'great vessel atherosclerosis',
                                  'h_great_vessel_calcification':'great vessel calcification',
                                  'h_great_vessel_catheter_or_port':'great vessel catheter or port',
                                  'h_great_vessel_dilation_or_ectasia':'great vessel dilation',
                                  'h_great_vessel_postsurgical':'great vessel postsurgical',
                                  'h_great_vessel_scattered_calc':'great vessel scattered calc',
                                  'h_mediastinum_calcification':'mediastinal calcification',
                                  'h_mediastinum_cancer':'mediastinal cancer',
                                  'h_mediastinum_lymphadenopathy':'mediastinal lymphadenopathy',
                                  'h_mediastinum_mass':'mediastinal mass',
                                  'h_mediastinum_nodule':'mediastinal nodule',
                                  'h_mediastinum_opacity':'mediastinal opacity'}

HEART_COLS = ['heart_atherosclerosis',
              'heart_coronary_artery_disease',
              'heart_calcification',
              'heart_scattered_calc',
              'heart_postsurgical',
              'heart_sternotomy',
              'heart_cabg',
              'heart_transplant',
              'heart_catheter_or_port','heart_stent',
              'heart_heart_valve_replacement', 'heart_pacemaker_or_defib',
              'heart_cardiomegaly', 
              'heart_heart_failure',
              'heart_pericardial_effusion', 'heart_pericardial_thickening']

HEART_COLS_RENAME = {'heart_atherosclerosis':'heart atherosclerosis',
                     'heart_cabg':'CABG',
                     'heart_calcification':'heart calcification',
                     'heart_cardiomegaly':'cardiomegaly',
                     'heart_catheter_or_port':'heart catheter or port',
                     'heart_coronary_artery_disease':'coronary artery disease',
                     'heart_heart_failure':'heart failure',
                     'heart_heart_valve_replacement':'valve replacement',
                     'heart_pacemaker_or_defib':'pacemaker or defibrillator',
                     'heart_pericardial_effusion':'pericardial effusion',
                     'heart_pericardial_thickening':'pericardial thickening',
                     'heart_postsurgical':'heart postsurgical',
                     'heart_scattered_calc':'heart scattered calcification',
                     'heart_stent':'heart stent',
                     'heart_sternotomy':'sternotomy',
                     'heart_transplant':'heart transplant'}

LUNG_COLS = [#Diffuse findings
             'lung_air_trapping',
             'lung_airspace_disease',
             'lung_emphysema',
             'lung_atelectasis',
             #Lung patterns
             'lung_bandlike_or_linear',
             'lung_groundglass', 
             'lung_reticulation',
             'lung_tree_in_bud',
             'lung_bronchial_wall_thickening',
             'lung_bronchiectasis',
             'lung_bronchiolectasis',
             'lung_calcification',
             #Pneumonia-like
             'lung_cavitation', 
             'lung_consolidation',
             'lung_aspiration',
             'lung_infection',
             'lung_pneumonia',
             'lung_inflammation',
             #ILD-like
             'lung_fibrosis',
             'lung_honeycombing',
             'lung_interstitial_lung_disease',
             #Pleura
             'lung_plaque',
             'lung_pleural_thickening',
             'lung_pleural_effusion',
             #Misc diffuse
             'lung_pneumothorax', 
             'lung_scattered_calc', 'lung_scattered_nod',
             'lung_septal_thickening', 'lung_soft_tissue',
             
             #Surgery
             'lung_postsurgical',
             'lung_lung_resection',
             'lung_transplant',
             'lung_pulmonary_edema',  'lung_scarring',
              
              #Human-made
             'lung_catheter_or_port',
             'lung_chest_tube','lung_clip',
             'lung_staple','lung_suture',
             
             #Focal findings:
             'lung_cancer','lung_cyst', 'lung_density','lung_granuloma',
             'lung_lesion', 'lung_lucency', 
             'lung_lymphadenopathy', 'lung_mass', 'lung_mucous_plugging',
             'lung_nodule', 'lung_nodulegr1cm', 'lung_opacity']

LUNG_COLS_RENAME = {'lung_air_trapping':'air trapping',
                    'lung_airspace_disease':'airspace disease',
                    'lung_aspiration':'aspiration',
                    'lung_atelectasis':'atelectasis',
                    'lung_bandlike_or_linear':'bandlike or linear',
                    'lung_bronchial_wall_thickening':'bronchial wall thickening',
                    'lung_bronchiectasis':'bronchiectasis',
                    'lung_bronchiolectasis':'bronchiolectasis',
                    'lung_calcification':'lung calcification',
                    'lung_cancer':'lung cancer',
                    'lung_catheter_or_port':'lung catheter or port',
                    'lung_cavitation':'cavitation',
                    'lung_chest_tube':'chest tube',
                    'lung_clip':'lung clip',
                    'lung_consolidation':'lung consolidation',
                    'lung_cyst':'lung cyst',
                    'lung_density':'lung density',
                    'lung_emphysema':'emphysema',
                    'lung_fibrosis':'fibrosis',
                    'lung_granuloma':'granuloma',
                    'lung_groundglass':'groundglass',
                    'lung_honeycombing':'honeycombing',
                    'lung_infection':'lung infection',
                    'lung_inflammation':'lung inflammation',
                    'lung_interstitial_lung_disease':'interstitial lung disease',
                    'lung_lesion':'lung lesion',
                    'lung_lucency':'lung lucency',
                    'lung_lung_resection':'lung resection',
                    'lung_lymphadenopathy':'lung lymphadenopathy',
                    'lung_mass':'lung mass',
                    'lung_mucous_plugging':'mucous plugging',
                    'lung_nodule':'lung nodule',
                    'lung_nodulegr1cm':'lung nodulegr1cm',
                    'lung_opacity':'lung opacity',
                    'lung_plaque':'lung plaque',
                    'lung_pleural_effusion':'pleural effusion',
                    'lung_pleural_thickening':'pleural thickening',
                    'lung_pneumonia':'pneumonia',
                    'lung_pneumothorax':'pneumothorax',
                    'lung_postsurgical':'lung postsurgical',
                    'lung_pulmonary_edema':'pulmonary edema',
                    'lung_reticulation':'reticulation',
                    'lung_scarring':'lung scarring',
                    'lung_scattered_calc':'lung scattered calcifications',
                    'lung_scattered_nod':'lung scattered nod',
                    'lung_septal_thickening':'septal thickening',
                    'lung_soft_tissue':'lung soft tissue',
                    'lung_staple':'lung staple',
                    'lung_suture':'lung suture',
                    'lung_transplant':'lung transplant',
                    'lung_tree_in_bud':'tree-in-bud'}

def return_renamers():
    return GV_AND_MEDIA_COLS, GV_AND_MEDIA_COLS_RENAME, HEART_COLS, HEART_COLS_RENAME, LUNG_COLS, LUNG_COLS_RENAME

    