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

def ratio_extractor (direct):

   os.chdir(direct)

   df_filenames = glob.glob('*analysis.hdf5' )

   df_total_list = []
   df_pos_list = [] 

   for n in range(len(df_filenames)):
   
     store = HDFStore(df_filenames[n])
     df_total = 0
     df_pos = 0  

     for key in store.keys():
        df = store[key]
        #df['radius_micron'] = df['radius'] * 0.325
        
        if df ['radius_micron'].tolist()[0] > 4 and df ['Eccentricity'].tolist()[0] <= 0.4 and df ['Eccentricity'].tolist()[0] != 0.0:
          df_pos += 1
        
        df_total +=1 
        
   
     df_total_list.append(df_total)
     df_pos_list.append(df_pos)

     store.close()

   ratio_list = np.array(df_pos_list) / np.array(df_total_list)
   
   return ratio_list

ratio_list_1 = ratio_extractor(root1.directory)


root2 = tk.Tk()
root2.withdraw()

GUV_Post_Analysis_df_list = []

my_filetypes = [('all files', '.*'),('Image files', '.hdf5')]

#filez = filedialog.askopenfilenames(parent = root, title='Please Select a File', filetypes = my_filetypes)
root2.update() # To prevent open file dialog freeze after selecting the file
root2.directory = filedialog.askdirectory(parent = root2)

ratio_list_2 = ratio_extractor(root2.directory)

print(scipy.stats.ttest_ind(ratio_list_1, ratio_list_2)) 

condition_1 = len(ratio_list_1) * ['Original Approach']
condition_2 = len(ratio_list_2) * ['Modified Approach']

ratio_df1 = pd.DataFrame({'ratio':ratio_list_1,'condition':condition_1})
ratio_df2 = pd.DataFrame({'ratio':ratio_list_2,'condition':condition_2})

ratio_df = pd.concat([ratio_df1,ratio_df2])

# Drawthe point plot with individual data points

fig, ax = plt.subplots()
fig.set_size_inches(12, 8)

sns.set(style="ticks")
sns.set_context("paper", font_scale=2, rc={"font.size":16,"axes.labelsize":16})


ax = sns.pointplot(x="condition", y="ratio", data=ratio_df, ci=95, color='k',capsize = 0.1, errwidth=4)
ax = sns.swarmplot(x="condition", y="ratio",data=ratio_df, hue = 'condition',size = 8,palette=['b','r'])


#labels = ['Hypotonic \n(n = %d)' % len(Hypo),'Isotonic \n(n = %d)' % len(Iso)]

#fig.set(xticklabels = labels)

ax.tick_params(labelsize= 16)
plt.xlabel('')
plt.ylabel('Percent of Unilamellar Vesicles', fontsize=14, fontweight = 'bold')
plt.rcParams['axes.labelweight'] = 'bold'
#plt.show()

# Save as PDF
fig_save_name = filedialog.asksaveasfilename(parent=root1,title="Please select a name for saving figure:",filetypes=[('Graph', '.pdf')])
plt.savefig(fig_save_name, transparent=True)
