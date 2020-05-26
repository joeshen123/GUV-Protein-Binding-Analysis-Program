import tkinter as tk
from tkinter import filedialog
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
import glob,os
import warnings
from pandas import HDFStore
import numpy as np
import h5py
from matplotlib import gridspec
from matplotlib.pyplot import cm
import mplcursors
# Make a function to combine all df together into one df. Input is the directory of all dfs
def curve_df_combine(directory):
   os.chdir(directory)

   df_list = []

   df_filenames = glob.glob('*analysis.hdf5' )


   for n in range(len(df_filenames)):
      df_name = df_filenames[n]
      store = HDFStore(df_name)

      for key in store.keys():
         df = store[key]

         #df['Normalized GFP intensity'] = df['Normalized GFP intensity'] /df['Normalized GFP intensity'][0]
         df_list.append(df)

      store.close()


   df_len = len(df_list)

   #df_final = pd.concat(df_list)

   return df_list, df_len

# Define a function to plot both radius and protein intensity from a list of pandas df
def Pandas_list_plotting(pandas_list, keyword, marker = None):
    fig= plt.figure(figsize = (10,6))
    gs = gridspec.GridSpec(1,1)

    ax = fig.add_subplot(gs[:,:])

    list_len = len(pandas_list)
    color = cm.tab20b(np.linspace(0,1,list_len))

    if keyword == 'Normalized Intensity':
      ax.set_title('Vesicles Protein Bindings Changes',fontsize=18)

      ax.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold')
      ax.set_ylabel('Protein Fluorescence Intensity', fontsize = 16,fontweight = 'bold')

      for n in range(list_len):
         df = pandas_list[n]
         Intensity_data = df['Normalized GFP intensity'].tolist()
         Time_point = df['Time Point'].tolist()
         ax = plt.plot(Time_point, Intensity_data, color = color[n], label = str(n), marker = marker)

    if keyword == 'Radius':
      ax.set_title('Vesicles Radius Changes (um)',fontsize=18)

      ax.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold')
      ax.set_ylabel('Vesicles Radius (um)', fontsize = 16,fontweight = 'bold')

      for n in range(list_len):
         df = pandas_list[n]
         radius_data = df['radius_micron'].tolist()
         Time_point = df['Time Point'].tolist()
         ax = plt.plot(Time_point, radius_data, color = color[n], label = str(n), marker = marker)


    #plt.legend(loc='right')
    plt.tight_layout()
    plt.ylim((0,1800))
    #mplcursors.cursor(highlight=True,hover=True)
    plt.show()

root = tk.Tk()
root.withdraw()


root.directory = filedialog.askdirectory()

Name1 = root.directory
label_1 = Name1.split("/")[-1]


df_final_list,df_final_one_len = curve_df_combine(Name1)


Pandas_list_plotting(df_final_list,'Normalized Intensity',marker='o')
#print(df_final_one)
