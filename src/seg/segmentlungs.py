#segmentlungs.py
#Heavily modified from https://www.kaggle.com/arturscussel/lung-segmentation-and-candidate-points-generation
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
import copy
import timeit
import numpy as np
import pandas as pd
from skimage import measure
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

import matplotlib
matplotlib.use('agg') #so that it does not attempt to display via SSH
import matplotlib.pyplot as plt
plt.ioff() #turn interactive plotting off

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class SegmentLung():
    def __init__(self,
                 volume_log_df_path,
                 ct_scan_path,
                 results_dir,
                 visualize):
        """Segment the lungs frm CT volumes.
        
        Variables:
        <volume_log_df_path>: path to a log file that defines what volumes will be
            processed. At a minimum must include columns
            ['VolumeAcc_DEID','NoteAcc_DEID','MRN_DEID','Set_Assigned','Subset_Assigned']
            Example: file originally used (has old PHI column names): '/home/rlb61/data/PreprocessVolumes/2019-10-30_volume_preprocessing/CT_Scan_Preprocessing_Log_File_FINAL_SMALL.csv'
            Example: updated DEID file to use: '/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/CT_Scan_Metadata_Small_DEID.csv'
        <ct_scan_path>: path to the raw CT scans.
            Example: path originally used (contains PHI-named CTs): '/scratch/rlb61/2019-10-BigData/'
            Example: updated DEID directory to use: '/scratch/rlb61/2019-10-BigData-DEID/'
        <results_dir>: directory to store results.
            Example: path originally used: '/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-segmaps/'
        <visualize>: Boolean. If True then print intermediate summary statements,
            save 3D plots, and save a ghost plot of the lungs throughout the
            segmentation process."""
        self.volume_log_df = pd.read_csv(volume_log_df_path, header=0,index_col='VolumeAcc_DEID')
        self.ct_scan_path = ct_scan_path
        self.results_dir = results_dir
        self.visualize = visualize
        
        #hu_thresh and vol_thresh were carefully calibrated in order to
        #obtain high-quality lung segmentations. Do NOT change these values!
        self.hu_thresh = -300 #integer for Hounsfield Unit cutoff that defines lung
        self.vol_thresh = 1000000 #minimum volume to keep in 3d cleaning step
        
        #Dataframe to save extreme points:
        self.extrema = copy.deepcopy(self.volume_log_df[['NoteAcc_DEID','MRN_DEID','Set_Assigned','Subset_Assigned']])
        for newcol in ['sup_axis0min','inf_axis0max','ant_axis1min','pos_axis1max',
                       'rig_axis2min','lef_axis2max','shape0','shape1','shape2']:
            self.extrema[newcol] = np.nan
        
        #Load anything that was previously done (overwrrite self.extrema):
        extremafile = os.path.join(self.results_dir,'extrema.csv')
        if os.path.isfile(extremafile):
            print('Loading from previous extrema file,',extremafile)
            self.extrema = pd.read_csv(extremafile, header=0, index_col=0)
        
        #Do segmentation and find extrema
        unfinished = self.extrema[self.extrema.isnull().any(axis=1)]
        print('Working on unfinished scans, count',unfinished.shape[0])
        for idx in range(unfinished.shape[0]):
            filename = unfinished.index.values.tolist()[idx]
            self.segment_from_2d(filename)
            if idx % 200 == 0: print('Finished',str(idx)+'/'+str(len(unfinished.index.values.tolist())),'=',100*round(idx/len(unfinished.index.values.tolist()),2),'percent')
    
    ###########
    # Methods #-----------------------------------------------------------------
    ###########
    def segment_from_2d(self, filename):
        """Segment <ctvol> one 2d slice at a time"""
        tot0 = timeit.default_timer()
        
        #Load CT volume
        ctvol = np.load(os.path.join(self.ct_scan_path, filename))['ct']
        
        if self.visualize:
            print(filename,'\n','\tPixel value range:',np.amin(ctvol),np.amax(ctvol))
            print('\tShape:',ctvol.shape)
            segmap_1 = np.zeros(ctvol.shape)
            segmap_2 = np.zeros(ctvol.shape)
        
        #Obtain initial lung segmentation based on an algorithm applied to each
        #2D slice of the volume
        #example ctvol shape (488, 512, 512)
        segmap_4 = np.zeros(ctvol.shape)
        for slice_idx in range(ctvol.shape[0]):
            binary_1, binary_2, binary_4 = self.segment_slice_2d(copy.deepcopy(ctvol[slice_idx,:,:]))
            
            #segmap_4 is the important one that will be used to make the final map
            segmap_4[slice_idx,:,:] = binary_4
            
            if self.visualize:
                #if we want to make visualizations then also save the results
                #of the intermediate steps
                segmap_1[slice_idx,:,:] = binary_1
                segmap_2[slice_idx,:,:] = binary_2
                if slice_idx % 100 == 0:
                    print('\t\tFinished slice',slice_idx)
        
        #Clean up volume in 3D (cleaning is VERY important):
        segmap_4 = segmap_4.astype('int')
        segmap_final = self.clean_up_volume_produced_from_2d(segmap_4)
        tot1 = timeit.default_timer()
        
        #Save the segmentation
        np.savez_compressed(os.path.join(self.results_dir, filename.replace('.npz','')+'_seg.npz'), segmap=segmap_final)
        
        #Find the extrema
        self.find_lung_extrema(segmap_final, filename)
        
        #Make visualizations if relevant
        if self.visualize:
            print('\t'+filename+' Total Seg Time', round((tot1 - tot0)/60.0,2),'minutes')
            self.plot_3D_segmap(segmap_1, filename, '1')
            self.plot_3D_segmap(segmap_2, filename, '2')
            self.plot_3D_segmap(segmap_4, filename, '4')
            self.plot_3D_segmap(segmap_final, filename, 'final')
    
    def segment_slice_2d(self, inputarray):
        """Segment the lungs from a 2D slice <inputarray>"""
        #Step 1: Convert into a binary image.
        binary_1 = inputarray < self.hu_thresh
        
        #Step 2: Remove the blobs connected to the border of the image.
        binary_2 = clear_border(binary_1)
    
        #Step 3: Label the image.
        label_image = label(binary_2)
        
        #Step 4: Keep the labels with 2 largest areas.
        areas = [r.area for r in regionprops(label_image)]
        areas.sort()
        if len(areas) > 2:
            for region in regionprops(label_image):
                if region.area < areas[-2]:
                    for coordinates in region.coords:                
                           label_image[coordinates[0], coordinates[1]] = 0
        binary_4 = label_image > 0
        return binary_1.astype('int'), binary_2.astype('int'), binary_4.astype('int')
    
    def clean_up_volume_produced_from_2d(self, segmap):
        """Clean up a 3D segmentation map that was produced using 2D slices"""
        #Note that it is critical to put in the segmap as type 'bool' so that
        #the small objects will actually be removed. If you put in the segmap
        #as type 'int' then it will assume everything that is a 1 is part of
        #the same object, and it won't remove anything!
        segmap = remove_small_objects(segmap.astype('bool'), min_size = self.vol_thresh)
        return segmap.astype('int')
    
    # Find Extreme Points ######################################################
    def find_lung_extrema(self, segmap, filename):
        """Save the superior, inferior, anterior, posterior, left, and right
        coordinates of the lung and if indicated save a 'ghost plot'"""
        #segmap is [slices, square, square]
        #Sum across axes:
        zflat = np.sum(segmap, axis=0) #two circles next to each other (lungs seen from the top)
        yflat = np.sum(segmap, axis=1) #two arches next to each other (lungs seen from the front)
        xflat = np.sum(segmap, axis=2) #one arch (lungs seen from the side)
        
        #Bounding box coordinates:
        label_image = label(segmap)
        rps = regionprops(label_image)
        if len(rps) == 1:
            bounds = rps[0].bbox
        else:
            if self.visualize: print('\tFor filename',filename,'the number of regions was',len(rps))
            bounds = [np.inf, np.inf, np.inf, -1*np.inf, -1*np.inf, -1*np.inf]
            for ridx in range(len(rps)):
                selected_bounds = rps[ridx].bbox
                if self.visualize: print('\t'+str(selected_bounds))
                for bidx in [0,1,2]: #minimums
                    if selected_bounds[bidx] < bounds[bidx]:
                        bounds[bidx] = selected_bounds[bidx]
                for bidx in [3,4,5]: #maximums
                    if selected_bounds[bidx] > bounds[bidx]:
                        bounds[bidx] = selected_bounds[bidx]
        
        lef = bounds[5]; rig = bounds[2]
        ant = bounds[1]; pos = bounds[4]
        inf = bounds[3]; sup = bounds[0] 
        self.extrema.at[filename, 'sup_axis0min'] = sup #0, zmin, axis0min
        self.extrema.at[filename, 'ant_axis1min'] = ant #1, ymin, axis1min
        self.extrema.at[filename, 'rig_axis2min'] = rig #2, xmin, axis2min
        self.extrema.at[filename, 'inf_axis0max'] = inf #3, zmax, axis0max
        self.extrema.at[filename, 'pos_axis1max'] = pos #4, ymax, axis1max
        self.extrema.at[filename, 'lef_axis2max'] = lef #5, xmax, axis2max
        self.extrema.at[filename, 'shape0'] = segmap.shape[0]
        self.extrema.at[filename, 'shape1'] = segmap.shape[1]
        self.extrema.at[filename, 'shape2'] = segmap.shape[2]
        
        if self.visualize:
            print('\t',self.extrema.loc[filename,:])
            self.make_ghost_plot(filename, segmap, zflat, yflat, xflat, ant, pos, lef, rig, inf, sup)
        
        #Save extrema
        self.extrema.to_csv(os.path.join(self.results_dir,'extrema.csv'))
    
    def make_ghost_plot(self, filename, segmap, zflat, yflat, xflat,
                        ant, pos, lef, rig, inf, sup):
        """Save 'ghost plot' in which segmentation map <segmap> is projected
        in three ways and the calculated bounding box is plotted on the figure
        to enable visual sanity checks."""
        figure, plots = plt.subplots(3, 1, figsize=(5, 15))
        
        plots[0].imshow(zflat, cmap=plt.cm.bone)
        plots[0].set_title('Flattened Z Axis (0)')
        plots[0].set_xlabel('X coordinate')
        plots[0].set_ylabel('Y coordinate')
        
        plots[1].imshow(yflat, cmap=plt.cm.bone)
        plots[1].set_title('Flattened Y Axis (1)')
        plots[1].set_xlabel('X coordinate')
        plots[1].set_ylabel('Z coordinate')
        
        plots[2].imshow(xflat, cmap=plt.cm.bone)
        plots[2].set_title('Flattened X Axis (2)')
        plots[2].set_xlabel('Y coordinate')
        plots[2].set_ylabel('Z coordinate')
        figure.subplots_adjust(hspace=0.25) #so the subplots are not so tightly packed
        figure.savefig(os.path.join(self.results_dir, filename.replace('.npz','')+'_GhostPlots.png'))
        
        #Now save a version with lines:
        plots[0].axhline(y=ant,color='paleturquoise')
        plots[0].text(x=segmap.shape[2]/2,y=ant-12,s='ant_axis1min='+str(ant),color='paleturquoise')
        plots[0].axhline(y=pos,color='paleturquoise')
        plots[0].text(x=segmap.shape[0]/2,y=pos+22,s='pos_axis1max='+str(pos),color='paleturquoise')
        
        plots[1].axvline(x=lef,color='paleturquoise')
        plots[1].text(x=lef+7,y=segmap.shape[0]/2,s='lef_axis2max='+str(lef),color='paleturquoise',rotation=90)
        plots[1].axvline(x=rig,color='paleturquoise')
        plots[1].text(x=rig-22,y=segmap.shape[0]/2,s='rig_axis2min='+str(rig),color='paleturquoise',rotation=90)
        
        plots[2].axhline(y=inf,color='paleturquoise')
        plots[2].text(x=segmap.shape[0]/2,y=inf+22,s='inf_axis0max='+str(inf),color='paleturquoise')
        plots[2].axhline(y=sup,color='paleturquoise')
        plots[2].text(x=segmap.shape[0]/2,y=sup-12,s='sup_axis0min='+str(sup),color='paleturquoise')
        
        figure.savefig(os.path.join(self.results_dir, filename.replace('.npz','')+'_GhostPlotsAnnotated.png'))
    
    # Orient and Plot 3D Segmap ################################################
    def orient_volume(self, volume):
        """Orient a volume so that the patient is facing you with their head
        towards the ceiling"""
        volume = volume.transpose(2,1,0)
        volume = np.flip(volume, axis=2)
        return volume
    
    def plot_3D_segmap(self, segmap, filename, descriptor):
        """Make a 3D plot of the binary segmentation map of the lungs.
        <filename> is the file name of the original CT volume which will be
            combined with <descriptor> to create the output file name."""
        print('\tPlot 3D Segmentation Map')
        #First orient volume
        print('Orienting segmap in order to make 3D plot')
        segmap = self.orient_volume(segmap)
        
        #from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
        #Get high threshold to show mostly bones
        threshold =0.5
        
        #https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.marching_cubes_lewiner
        verts, faces, _ignore1, _ignore2 = measure.marching_cubes_lewiner(segmap, threshold)
    
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces], alpha=0.70)
        face_color = [0.45, 0.45, 0.75]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    
        ax.set_xlim(0, segmap.shape[0])
        ax.set_xlabel('x axis (0)')
        ax.set_ylim(0, segmap.shape[1])
        ax.set_ylabel('y axis (1)')
        ax.set_zlim(0, segmap.shape[2])
        ax.set_zlabel('z axis (2)')
        plt.title('3D Segmentation '+str(self.hu_thresh))
        
        #must save as a png because Firefox can't render the pdfs
        plt.savefig(os.path.join(self.results_dir, filename.replace('.npz','')+'_'+descriptor+'_3DLungPlot.png'))
        plt.close()   
    