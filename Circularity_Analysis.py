import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import warnings
from tkinter import simpledialog
from skimage import measure
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pandas import HDFStore
import pandas as  pd

#Ignore warnings issued by skimage through conversion to uint8
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)
warnings.simplefilter("ignore",FutureWarning)
# Use tkinter to interactively select files to import
root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('Image files', '.hdf5')]

filez = filedialog.askopenfilenames(parent = root, title='Please Select a File', filetypes = my_filetypes)

file_name= root.tk.splitlist(filez)[0]

f = h5py.File(file_name, 'r')

my_filetypes = [('all files', '.*'),('Image files', '.hdf5')]

filez = filedialog.askopenfilenames(parent = root, title='Please Select a File', filetypes = my_filetypes)

file_name1= root.tk.splitlist(filez)[0]



#root.destroy()
store = HDFStore(file_name1)

df_list = []
for key in store.keys():
    print(key)
    df = store[key]
    df_list.append(df)

    store.close()


df = df_list[0]

Time = df['Time Point'].values

Time = np.linspace(0, 35, num = len(Time))

GFP_Intensity = df['Normalized GFP intensity'].values


binary_image = f['Segmentation_Binary_Result'][:]

binary_image_max= np.max(binary_image, axis=1)

circularity_list = []
for im in binary_image_max:
    region = measure.regionprops(measure.label(im))
    eccentricity = region[0].eccentricity
    
    circularity_list.append(1-eccentricity)

fig, ax1 = plt.subplots(figsize=(16, 12))

ax1.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold',color = 'k')
#ax1.set_ylabel('Cpla2 C2 Bindings', fontsize = 16,fontweight = 'bold',color ='r')
ax1.set_ylabel('Circularity', fontsize = 16,fontweight = 'bold',color = 'r')

plt1 = ax1.plot(Time, circularity_list,'ro-', markersize=10, linewidth=6)
#plt1 = ax1.plot(Time, radius, 'b-')

ax1.tick_params(axis = 'y', labelcolor = 'r',labelsize = 'x-large')
ax1.tick_params(axis = 'x',  labelsize = 'x-large',labelcolor = 'k')

ax2 = ax1.twinx()
#x2.set_ylim((80,150))
ax2.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold')
ax2.set_ylabel('cPla2 C2 Binding Intensities', fontsize = 16,fontweight = 'bold',color='b')
plt2 = ax2.plot(Time, GFP_Intensity, 'bo-', markersize=10, linewidth=6)
ax2.tick_params(axis = 'y', labelcolor = 'b',labelsize = 'x-large')

plt_total = plt2 + plt1

fig.tight_layout()
#plt.show()

fig_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving figure:",filetypes=[('Graph', '.tiff')])
plt.savefig(fig_save_name, dpi=300)