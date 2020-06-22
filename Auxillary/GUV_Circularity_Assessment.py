import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import warnings
from tkinter import simpledialog
from tkinter import Tk, Label, Button, Radiobutton, IntVar
import h5py
import numpy as np
from skimage import transform
from scipy.ndimage import zoom
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
import napari
import time
import matplotlib
from pandas import HDFStore
import os
import glob
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
matplotlib.use('Qt4Agg')

# Save specific font that can be recognized by Adobe Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#Ignore warnings issued by skimage through conversion to uint8
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)
warnings.simplefilter("ignore",FutureWarning)
# Use tkinter to interactively select files to import
root1 = tk.Tk()
root1.withdraw()

GUV_Post_Analysis_df_list = []

#my_filetypes = [('all files', '.*'),('Image files', '.hdf5')]

#filez = filedialog.askopenfilenames(parent = root1, title='Please Select a File', filetypes = my_filetypes)
root1.update() # To prevent open file dialog freeze after selecting the file
root1.directory = filedialog.askdirectory(parent = root1)
'''
df_name= root1.tk.splitlist(filez)[0]
#print(df_name)

f = h5py.File(df_name, 'r')


seg_image = f['Segmentation_Binary_Result'][0]

region = regionprops(label(seg_image))

for r in region:
    print(r.area)
#seg_image = label(seg_image)
with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_labels(label(seg_image), name='seg_image',blending = 'opaque', scale = (4,1,1))
'''


#for n in range(len(df_filenames)):
      #df_name = df_filenames[n]
      #print(df_name)

def eccen_extractor (direct):

   os.chdir(direct)

   df_filenames = glob.glob('*_seg.npy' )

   eccen_list = [] 

   for n in tqdm(range(len(df_filenames))):
   
     image_data = np.load(df_filenames[n], allow_pickle=True).item()
     masks = image_data['masks']
     masks = clear_border(masks)

     labelled = label(masks)
     
     region = regionprops(labelled)
     
     eccen_list_mean = []
     for r in region:
        eccen = 1-r.eccentricity
        eccen_list_mean.append(eccen)
     
     mean = np.mean(np.array(eccen_list_mean))

     eccen_list.append(mean)

   
   return np.array(eccen_list)

eccen_list_1 = eccen_extractor(root1.directory)


root2 = tk.Tk()
root2.withdraw()

GUV_Post_Analysis_df_list = []

my_filetypes = [('all files', '.*'),('Image files', '.hdf5')]

#filez = filedialog.askopenfilenames(parent = root, title='Please Select a File', filetypes = my_filetypes)
root2.update() # To prevent open file dialog freeze after selecting the file
root2.directory = filedialog.askdirectory(parent = root2)

eccen_list_2 = eccen_extractor(root2.directory)

print(scipy.stats.ttest_ind(eccen_list_1, eccen_list_2, equal_var=False)) 

condition_1 = len(eccen_list_1) * ['Original Approach']
condition_2 = len(eccen_list_2) * ['Modified Approach']

eccen_df1 = pd.DataFrame({'Circularity':eccen_list_1,'condition':condition_1})
eccen_df2 = pd.DataFrame({'Circularity':eccen_list_2,'condition':condition_2})

ratio_df = pd.concat([eccen_df1,eccen_df2])

# Drawthe point plot with individual data points

fig, ax = plt.subplots()
fig.set_size_inches(12, 8)

sns.set(style="ticks")
sns.set_context("paper", font_scale=2, rc={"font.size":16,"axes.labelsize":16})


ax = sns.pointplot(x="condition", y="Circularity", data=ratio_df, ci=95, color='k',capsize = 0.1, errwidth=4)
ax = sns.swarmplot(x="condition", y="Circularity",data=ratio_df, hue = 'condition',size = 8,palette=['b','r'])

ax.set_ylim([0.45,0.8])
#labels = ['Hypotonic \n(n = %d)' % len(Hypo),'Isotonic \n(n = %d)' % len(Iso)]

#fig.set(xticklabels = labels)

ax.tick_params(labelsize= 16)
plt.xlabel('')
plt.ylabel('Circularity', fontsize=14, fontweight = 'bold')
plt.rcParams['axes.labelweight'] = 'bold'
#plt.show()

# Save as PDF
fig_save_name = filedialog.asksaveasfilename(parent=root1,title="Please select a name for saving figure:",filetypes=[('Graph', '.pdf')])
plt.savefig(fig_save_name, transparent=True)
