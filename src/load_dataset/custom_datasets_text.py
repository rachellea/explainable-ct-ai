#custom_datasets_text.py
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

import torch
import string
import random
import numpy as np

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
#plt.ioff()
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.load_dataset import custom_datasets
from src.load_dataset.vol_proc import ctvol_preproc

class CTDataset_2019_10_with_Text(custom_datasets.CTDataset_2019_10):
    """Augments the CT dataset by (a) putting text randomly on some slices
    of the CT volume and (b) creating a 'textlabel' which is a binary vector
    with length equal to the number of slices, where an element equals 1 if
    there is text present on that slice and is equal to 0 otherwise."""
    
    def __getitem__(self, idx):
        """Same as __getitem__ in CTDataset_2019_10 except that the 'data' has
        had text added to it by the modified version of _prepare_this_ctvol(),
        and here we add a new dictionary key 'textlabel' which is a binary
        vector of ground truth for whether text has been added or not."""
        #Make sure we are using 'cube' data because we want full data aug
        assert self.crop_magnitude == 'cube'
        
        base_sample = self._get_pace(self.volume_accessions[idx], self.note_accessions[idx])
        base_sample['textlabel'] =  self.textlabel
        return base_sample
    
    def _prepare_this_ctvol(self, ctvol, volume_acc):
        """Major preprocessing on the <ctvol>"""
        data, auglabel, randpad6val = ctvol_preproc.prepare_ctvol_2019_10_dataset(ctvol = ctvol,
                                                   volume_acc = volume_acc,
                                                   extrema = self.extrema,
                                                   data_augment = self.data_augment,
                                                   **self.volume_prep_args)
        
        #Add text data to the ctvol and also define the textlabel
        data = self._add_text_aspects_to_ctvol(data)
        
        #Must return only these 3 things because that's what _prepare_this_ctvol
        #needs to return
        return data, auglabel, randpad6val
    
    ########################
    # Text-Related Methods #----------------------------------------------------
    ########################
    def _add_text_aspects_to_ctvol(self, data): #Tested by visualization. See test_custom_datasets_text.make_viz_for_text_over_ct_slice()
        """Modify the <data> so that the slices stack up along axis 0
        have had text added. Also create 'textlabel' binary label vector.
        
        <data> is a torch Tensor with shape approx. [405, 420, 420]. It has
            already been normalized so the values are floats between roughly
            -0.5 and 0.5 (it's not exact because the normalization was done
            based on fixed HU of e.g. -1000 to +800)"""
        data_shape = list(data.size()) #e.g. [405, 420, 420]
        
        #Initialize the textlabel with all 0s
        self.textlabel = np.zeros((data_shape[0]))
        
        #Modify the torch Tensor data so that each 0th-dimension 2d slice
        #contains text with a 50% probability
        for idx_axis0 in range(data_shape[0]):
            #with 50% probability, add text to the slice
            if random.randint(0,100) < 50:
                #Get a np array where the 1s indicate text
                text_array = return_np_array_containing_text(tuple(data_shape[1:]))
                
                #Insert text_array into the relevant idx_axis0 of a 3d array
                #so it can be used directly for masked fill
                text_array_3d = np.full(data_shape, False, dtype=bool)
                text_array_3d[idx_axis0,:,:] = text_array
                
                #Make sure text is actually present before you modify the
                #textlabel and the data (because of the randomness in generating
                #the np array with text, it's possible that no text actually
                #made it in, so that's why you need to check.)
                if np.sum(text_array_3d) > 0: #if greater than 0, text is present
                    self.textlabel[idx_axis0] = 1
                    #print(idx_axis0,'text_array_3d sum',np.sum(text_array_3d))
                                
                    #Choose grey values for the text, based on randomly sampling floats
                    #within the upper half of the Tensor's values:
                    max_val = torch.max(data).data.numpy().tolist()
                    greyvalue = random.uniform(int(max_val/2.0),max_val)
                     
                    #Modify the slice so that the area corresponding to text is
                    #shown in grey. Note that text_array_3d needs to be Boolean
                    #data type otherwise no modification of the data Tensor will take
                    #place.
                    data = data.masked_fill_(torch.from_numpy(text_array_3d), greyvalue)
                    
                    #Uncomment this to generate mask figures for sanity checks:
                    # if idx_axis0 in range(200,250):
                    #     plt.figure(figsize=(8, 8))
                    #     plt.imshow(text_array.astype('int'), cmap = plt.cm.gray)
                    #     plt.tight_layout(pad=0)
                    #     plt.gca().set_axis_off()
                    #     plt.savefig('mask_slice_'+str(idx_axis0)+'.png', bbox_inches='tight',pad_inches=0)
                    #     plt.close()
        return data
    
    def return_label_meanings(self):
        return ['slice_'+str(x) for x in range(self.volume_prep_args['max_slices'])]
    
def return_np_array_containing_text(shape_tuple): #Tested by visualization. See test_custom_datasets_text.make_viz_for_text_over_ct_slice()
    """Return a 2D Boolean numpy array (containing only 0 and 1) where the numpy
    array contains random text somewhere in it, as defined by the position
    of the Trues.
    It was really tricky to figure out how to write this without saving the
    array to disk as an intermediate step.
    
    <shape_tuple> a tuple of ints that defines the shape of the 2D array
        returned."""
    fig, ax = plt.subplots()
    plt.imshow(np.ones(shape_tuple),cmap=plt.cm.gray)
    
    #Randomize the text itself
    rand_text = return_random_text()
    
    #Randomize the font options
    #font options: https://matplotlib.org/3.3.0/tutorials/text/text_props.html
    rand_fontname = random.choice(['serif','sans-serif','monospace']) #'fantasy' and 'cursive' font families not found
    rand_fontsize = random.randint(10,30)
    rand_style = random.choice(['normal','italic','oblique'])
    rand_variant = random.choice(['normal','small-caps'])
    rand_stretch = random.choice(['ultra-condensed','extra-condensed','condensed','semi-condensed','normal','semi-expanded','expanded','extra-expanded','ultra-expanded'])
    rand_weight = random.choice(['ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'])
    
    #Randomize the position #TODO make sure this is right!!!! and you don't need to swap them!!!
    rand_horiz_coord = random.randint(0,shape_tuple[1]) #1 is columns; navigating columns moves horizontally
    rand_vert_coord = random.randint(0,shape_tuple[0]) #0 is rows; navigating rows moves vertically
    
    #Finally, put the randomized text in the image
    plt.text(rand_horiz_coord, rand_vert_coord,
             rand_text,
             color='white', #white for now; the color will be randomized when the font is transferred to the CT tensor
             fontname = rand_fontname,
             fontsize=rand_fontsize,
             style=rand_style,
             variant=rand_variant,
             stretch=rand_stretch,
             weight=rand_weight)
    
    #https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image-in-matplotlib 
    plt.gca().set_axis_off()
    
    # https://matplotlib.org/3.3.3/tutorials/intermediate/tight_layout_guide.html 
    plt.tight_layout(pad=0)
    
    #https://matplotlib.org/3.1.1/gallery/user_interfaces/canvasagg.html
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    
    #It is SUPER important to close the plot! If you don't then it messes up
    #the borders of subsequent plots.
    plt.close()
    
    #https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/
    #The average method simply averages the values: (R + G + B) / 3.
    X.shape
    X_blackwhite = np.average(X,axis=2)
    
    #Now set everything greater than or equal to 70 to 255, and everything below 70 to 0
    super_threshold_indices = X_blackwhite >= 70
    X_blackwhite[super_threshold_indices] = 255 #cannot set to 1 here, or in the next step it'll turn into zero!
    sub_threshold_indices = X_blackwhite < 70
    X_blackwhite[sub_threshold_indices] = 0
    
    #Now make into a boolean array
    X_blackwhite = (X_blackwhite > 0).astype('bool')
    
    #Now crop back to original shape
    axis_0_chop_first = int((X_blackwhite.shape[0]-shape_tuple[0])/2)
    axis_0_chop_last =   X_blackwhite.shape[0] - ((X_blackwhite.shape[0]-axis_0_chop_first) - shape_tuple[0])
    axis_1_chop_first = int((X_blackwhite.shape[1]-shape_tuple[1])/2)
    axis_1_chop_last =   X_blackwhite.shape[1] - ((X_blackwhite.shape[1]-axis_1_chop_first) - shape_tuple[1])
    X_blackwhite_chopped = X_blackwhite[axis_0_chop_first:axis_0_chop_last, axis_1_chop_first:axis_1_chop_last]
    assert (X_blackwhite_chopped.shape[0],X_blackwhite_chopped.shape[1])==shape_tuple, str((X_blackwhite_chopped.shape[0],X_blackwhite_chopped.shape[1]))+' neq '+str(shape_tuple)
    
    #And, finally, get rid of any leftover frame of ones
    X_blackwhite_final = remove_residual_frame_of_ones(X_blackwhite_chopped)
    
    return X_blackwhite_final

def remove_residual_frame_of_ones(np_array): #Done with testing
    """<np_array> is a Boolean array; remove rows of all Trues or columns of
    all Trues.
    Why is this function needed? For some reason, matplotlib is very insistent
    on being clever with borders even when I have tried and tried and tried to
    GET RID OF THE FRAME. This function will remove any residual borders that
    are stubbornly remaining even though I want them gone.
    (If I keep the borders around, a model could get OK performance by looking
    for the borders and ignoring the text, which would be very bad.)"""
    #We can detect rows of all ones if the sum of the row is equal to the
    #total number of columns:
    bad_rows = (np_array.sum(axis=1)) == np_array.shape[1]
    
    #We can detect columns of all ones if the sum of the column is equal to
    #the total number of rows
    bad_cols = (np_array.sum(axis=0)) == np_array.shape[0]
    
    #Note that we need to detect the bad rows and bad columns before fixing them
    #because the act of fixing them will mess up the above sum assumptions.
    #Fix the bad rows:
    for row_idx in range(np_array.shape[0]):
        #You MUST cast bad_rows[row_idx] to a bool explicitely using
        #bool(bad_rows[row_idx]) because otherwise
        #it will be of type <class 'numpy.bool_'> which means the 'is True'
        #part won't work, since 'True' is a generic Python bool!!!
        if bool(bad_rows[row_idx]) is True:
            np_array[row_idx,:] = False
    
    for col_idx in range(np_array.shape[1]):
        if bool(bad_cols[col_idx]) is True:
            np_array[:,col_idx] = False
        
    return np_array

def return_random_text():
    """With 50% probability, return a random string with random length
    between 0 and 20; otherwise return a string that is formatted to
    look like PHI."""
    if random.randint(0,100) < 50:
        #return random string composed of digits, ascii_letters, punctuation,
        #and/or whitespace. Exclude dollars signs and single quotes because they
        #may confuse matplotlib's text plotting functionality
        chars = string.printable.replace('$','').replace("'",'')
        return random_string(chars, random.randint(0,20))
    else:
        #return PHI-like string
        chosenrand = random.randint(0,100)
        if chosenrand < 33: #fake Firstname Lastname only
            return fake_firstname_lastname()
        elif 33 <= chosenrand <= 66: #fake MRN/accession only
            return fake_mrn_or_accession()
        else: #both fake name and fake MRN/accession
            return fake_firstname_lastname()+' '+fake_mrn_or_accession()
    
def fake_firstname_lastname():
    firstname = (random_string(string.ascii_lowercase, random.randint(0,10))).capitalize()
    lastname = (random_string(string.ascii_lowercase, random.randint(0,10))).capitalize()
    return (firstname+' '+lastname)

def fake_mrn_or_accession():
    letter_prefix = random_string(string.ascii_uppercase, random.randint(2,4))
    numeric_part = random_string(string.digits, random.randint(5,8))
    return (letter_prefix+numeric_part)

def random_string(character_set, chosen_length):
    """Return a random string based on the provided string <character_set> and
    int <chosen_length>"""
    return ''.join(random.choice(character_set) for i in range(chosen_length))

#######################
# For Deployment Only #---------------------------------------------------------
#######################
class CTDataset_2019_10_for_Text_Deployment(custom_datasets.CTDataset_2019_10):
    """Loads whole CT volumes with no cropping or padding. The 'textlabel'
    here is just a dummy label of all zeros, equal to the number of
    slices in the CT volume (possibly after it has been rotated into a
    coronal or sagittal view). The purpose of this dataset is to enable
    running the text classifier models on whole CT volumes."""
    def __getitem__(self, idx):
        base_sample = self._get_pace(self.volume_accessions[idx], self.note_accessions[idx])        
        base_sample['textlabel'] = torch.zeros(list(base_sample['data'].shape)[0])
        return base_sample
    
    def _prepare_this_ctvol(self, ctvol, volume_acc):
        #No data augmentation
        auglabel = [0,0,0,0,0,0]; randpad6val=[0,0,0,0,0,0]
        #The only processing is to normalize and center the pixel values and
        #convert to a torch Tensor
        ctvol = ctvol_preproc.torchify_pixelnorm_pixelcenter(ctvol, self.volume_prep_args['pixel_bounds'])
        #Must return only these 3 things because that's what _prepare_this_ctvol
        #needs to return
        return ctvol, auglabel, randpad6val
    
    def return_label_meanings(self):
        #assume the maximum possible number of label meanings, which is 958
        return ['slice_'+str(x) for x in range(958)]
