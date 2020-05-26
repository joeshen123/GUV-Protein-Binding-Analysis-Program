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
import matplotlib
import scipy
from GUV_Analysis_Module import *
# Save specific font that can be recognized by Adobe Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


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
         #print(len(df))
         #df = df[0:60]
         print(len(df))
         #df['Time Point'] = np.linspace(0, 60, num = 60)
         df_list.append(df)

      store.close()


   #df_final = pd.concat(df_list)

   return df_list

root = tk.Tk()
root.withdraw()


root.directory = filedialog.askdirectory(parent = root,initialdir=os.path.dirname(os.getcwd()))

Name1 = root.directory
label_1 = Name1.split("/")[-1]


df_final_one = curve_df_combine(Name1)

Pandas_list_plotting(df_final_one, 'Normalized Intensity',marker='o')
Pandas_list_plotting(df_final_one,'Radius',marker='o')



