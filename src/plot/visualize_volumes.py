#visualize_volumes.py
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
import imageio
import numpy as np
import ipywidgets as ipyw
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib
matplotlib.use('agg') #TODO uncomment if running ImageSliceViewer3D
import matplotlib.pyplot as plt
plt.ioff() #TODO uncomment if running ImageSliceViewer3D

try:
    import ctvol_preproc
except:
    print('Could not import ctvol_preproc')

class ImageSliceViewer3D:
    #Copied from https://github.com/mohakpatel/ImageSliceViewer3D
    """ 
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks. 
    
    User can interactively change the slice plane selection for the image and 
    the slice plane being viewed. 

    Argumentss:
    Volume = 3D input image (3D numpy array)
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find 
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html"""
    def __init__(self, volume, figsize=(8,8)):
        self.volume = volume
        self.figsize = figsize
        self.cmap = plt.cm.gray #alternative: cmap='plasma'
        self.v = [np.min(volume), np.max(volume)]
        
        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
                        options=['x-y','y-z', 'z-x'], value='x-y', 
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))
    
    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z":[1,2,0], "z-x":[2,0,1], "x-y": [0,1,2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1
        
        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice, 
            z=ipyw.IntSlider(min=0, max=maxZ, step=1, continuous_update=False, 
            description='Image Slice:'))
        
    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(self.vol[:,:,z], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v[0], vmax=self.v[1])

def visualize_crops(filename, results_dir, crop_approach, main_clean_path):
    """Make a 15-panel figure of 3 vertical crops x (top, middle, bottom)
    showing slices out of each crop.
    if <crop_approach> =='ctvol_preproc-fixed' then use fixed crop from ctvol_preproc
                       =='ctvol_preproc-aug' thenuse randomized crop from ctvol_preproc
    <main_clean_path> is the path to the CT scan directory for example
        /scratch/rlb61/2019-10-BigData-DEID"""
    ctvol = np.load(os.path.join(main_clean_path,filename))['ct'] #[slices, square, square]
    if crop_approach == 'ctvol_preproc-fixed':
        ctvol_dict = ctvol_preproc.triple_crop_3d_fixed(ctvol, max_slices=402, max_side_length=308)
    elif crop_approach == 'ctvol_preproc-aug':
        ctvol_dict = ctvol_preproc.triple_crop_3d_augment(ctvol, max_slices=402, max_side_length=308)
    visualize_crops_helper(filename, results_dir, ctvol_dict)
    
def visualize_crops_helper(filename, results_dir, ctvol_dict):
    """Helper for visualize_crops() which plots the crops specified in
    <ctvol_dict>"""
    #https://stackoverflow.com/questions/31726643/how-do-i-get-multiple-subplots-in-matplotlib
    #https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/figure_title.html
    fig, ax = plt.subplots(nrows = 3, ncols = 7)
    #Turn all axis ticks and ticklabels off (but keep axis labels)
    for row in range(3):
        for col in range(7):
            ax[row,col].tick_params(axis=u'both', which=u'both',length=0)
            ax[row,col].set_yticklabels([])
            ax[row,col].set_xticklabels([])
            
    keys = ['left_lung','heart','right_lung']
    for idx in range(len(keys)):
        key = keys[idx]
        ctvol = ctvol_dict[key]
        #Plot
        sixth = int(round(ctvol.shape[0]/6)) #e.g. 402/6 = 67
        ax[idx,0].set_ylabel(key)
        ax[idx,0].imshow(ctvol[0,:,:], cmap = plt.cm.gray)
        ax[idx,1].imshow(ctvol[1*sixth,:,:], cmap = plt.cm.gray) #e.g. 67
        ax[idx,2].imshow(ctvol[2*sixth,:,:], cmap = plt.cm.gray) #e.g. 134
        ax[idx,3].imshow(ctvol[3*sixth,:,:], cmap = plt.cm.gray) #e.g. 201
        ax[idx,4].imshow(ctvol[4*sixth,:,:] , cmap = plt.cm.gray) #e.g. 268
        ax[idx,5].imshow(ctvol[5*sixth,:,:], cmap = plt.cm.gray) #e.g. 335
        ax[idx,6].imshow(ctvol[-1,:,:], cmap = plt.cm.gray)
    fig.suptitle('top, mid, bot', fontsize=16)
    plt.savefig(os.path.join(results_dir, filename.replace('.npz','')+'_Crops.png'))

def plot_hu_histogram(ctvol, outprefix):
    #Histogram of HUs https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    plt.hist(ctvol.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.savefig(outprefix+'_HU_Hist.png')
    plt.close()

def plot_middle_slice(ctvol, outprefix):
    # Show some slice in the middle
    plt.imshow(ctvol[100,:,:], cmap=plt.cm.gray)
    plt.savefig(outprefix+'_Middle_Slice.png')
    plt.close()
    
def save_pngs_of_all_slices(ct_scan_path, filename, results_dir):
    ctvol = np.load(os.path.join(ct_scan_path,filename))['ct'] #[slices, square, square]
    for slice_num in range(ctvol.shape[0]):
        plt.imshow(ctvol[slice_num,:,:], cmap=plt.cm.gray)
        plt.savefig(os.path.join(results_dir,filename.replace('.npz','_'+str(slice_num)+'.png')))
        plt.close()

def plot_3d_skeleton(ctvol, units, outprefix):
    """Make a 3D plot of the skeleton.
    <units> either 'HU' or 'processed' (normalized) determines the thresholds"""
    #from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    #Get high threshold to show mostly bones
    if units == 'HU':
        threshold = 400
    elif units == 'processed':
        threshold = 0.99
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = ctvol.transpose(2,1,0)
    p = np.flip(p, axis = 0) #need this line or else the patient is upside-down
    
    #https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.marching_cubes_lewiner
    verts, faces, _ignore1, _ignore2 = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
    plt.savefig(outprefix+'_3D_Bones.png')
    plt.close()    

def make_gifs(ctvol, outprefix, chosen_views):
    """Save GIFs of the <ctvol> in the axial, sagittal, and coronal planes.
    This assumes the final orientation produced by the preprocess_volumes.py
    script: [slices, square, square].
    
    <chosen_views> is a list of strings that can contain any or all of
        ['axial','coronal','sagittal']. It specifies which view(s) will be
        made into gifs."""
    #https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
    
    #First fix the grayscale colors.
    #imageio assumes only 256 colors (uint8): https://stackoverflow.com/questions/41084883/imageio-how-to-increase-quality-of-output-gifs
    #If you do not truncate to a 256 range, imageio will do so on a per-slice
    #basis, which creates weird brightening and darkening artefacts in the gif.
    #Thus, before making the gif, you should truncate to the 0-256 range
    #and cast to a uint8 (the dtype imageio requires):
    #how to truncate to 0-256 range: https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    ctvol = np.clip(ctvol, a_min=-800, a_max=400)
    ctvol = (  ((ctvol+800)*(255))/(400+800)  ).astype('uint8')
    
    #Now create the gifs in each plane
    if 'axial' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[0]):
            images.append(ctvol[slicenum,:,:])
        imageio.mimsave(outprefix+'_axial.gif',images)
        print('\t\tdone with axial gif')
    
    if 'coronal' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[1]):
            images.append(ctvol[:,slicenum,:])
        imageio.mimsave(outprefix+'_coronal.gif',images)
        print('\t\tdone with coronal gif')
    
    if 'sagittal' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[2]):
            images.append(ctvol[:,:,slicenum])
        imageio.mimsave(outprefix+'_sagittal.gif',images)
        print('\t\tdone with sagittal gif')

if __name__ == '__main__':
    #Note - if you are going to run this code, you should updates the paths below
    
    # filenames =  [] #empty list in the repo because filenames are PHI
    # results_dir_fixed = '/home/rlb61/data/img-hiermodel2/results/2019-12-05-visualize-crops-fixed'
    # results_dir_aug = '/home/rlb61/data/img-hiermodel2/results/2019-12-05-visualize-crops-aug'
    # for results_dir in [results_dir_fixed, results_dir_aug]:
    #     if not os.path.exists(results_dir):
    #         os.mkdir(results_dir)
    # 
    # for filename in filenames:
    #     print('working on',filename)
    #     visualize_crops(filename, results_dir_fixed, crop_approach = 'ctvol_preproc-fixed')
    #     visualize_crops(filename, results_dir_aug, crop_approach = 'ctvol_preproc-aug')
        
    pass
