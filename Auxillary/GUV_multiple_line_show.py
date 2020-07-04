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
from tkinter import filedialog
from tkinter import messagebox
import warnings
from tkinter import simpledialog
from tkinter import Tk, Label, Button, Radiobutton, IntVar
# Save specific font that can be recognized by Adobe Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

my_filetypes = [('all files', '.*'),('Image files', '.hdf5')]
# Make a function to combine all df together into one df. Input is the directory of all dfs
def curve_df_combine(file_name):
   store = HDFStore(file_name)
   df_list = []
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


filez = filedialog.askopenfilenames(parent = root, title='Please Select a File', filetypes = my_filetypes)
root.update() # To prevent open file dialog freeze after selecting the file
file_name= root.tk.splitlist(filez)[0]
print(file_name)
df_final_one = curve_df_combine(file_name)

Pandas_list_plotting(df_final_one, 'Normalized Intensity',marker='o')
Pandas_list_plotting(df_final_one,'Radius',marker='o')

del_answer = messagebox.askyesnocancel("Question","Do you want to delete some measurements?")

while del_answer == True:
        delete_answer = simpledialog.askinteger("Input", "What number do you want to delete? ",
                                                 parent=root,
                                                 minvalue=0, maxvalue=100)


        df_final_one.pop(delete_answer)
        Pandas_list_plotting(df_final_one, 'Normalized Intensity',marker='o')
        Pandas_list_plotting(df_final_one,'Radius',marker='o')
        del_answer = messagebox.askyesnocancel("Question","Do you want to delete more measurements?")



File_save_names = '.'.join(file_name.split(".")[:-1])
list_save_name='{File_Name}_analysis'.format(File_Name = File_save_names)

#save as hdf5 file
n = 1
save_name = list_save_name +'.hdf5'
for df in df_final_one:
    key_tag = 'df_' + str(n)
    df.to_hdf(save_name, key = key_tag, complib='zlib', complevel=5)
    n += 1

# save as csv
csv_save_name = list_save_name + '.csv'
pd.concat(df_final_one).to_csv(csv_save_name)


