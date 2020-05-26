import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os,glob
from pandas import HDFStore
from tqdm import tqdm
import h5py
import scipy.stats as st
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib
import ast
# Save specific font that can be recognized by Adobe Illustrator
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype'] = 42
'''
# Make a function combine all df together into one df. Input is the directory of all dfs
def curve_df_combine(directory,num):
   os.chdir(directory)
   GFP_radi = []
   df_filenames = glob.glob('*analysis.hdf5' )
   #print(df_filenames)
   for n in tqdm(range(len(df_filenames))):
      df_name = df_filenames[n]
      store = HDFStore(df_name)
      for key in store.keys():
         df = store[key]
         print(df["radius"])
         radius_micron = (df['radius'].tolist())[num] * 0.33
         Fluo_data = (df['Normalized GFP intensity'].tolist())[num]
         
         radi_fluo = (radius_micron,Fluo_data)
         GFP_radi.append(radi_fluo)
      store.close()
   radi_fluo = np.array(radi_fluo)
   return GFP_radi

root = tk.Tk()
root.withdraw()
root.directory = filedialog.askdirectory()
Name1 = root.directory
label_1 = Name1.split("/")[-1]
'''
'''
r_fluo_dict = {}

for n in range(1):
   fluo_r = curve_df_combine(Name1,n)
   r_fluo_dict[n] = fluo_r
'''

'''
r_fluo_dict = curve_df_combine(Name1,0)
File_save_names = '.'.join(Name1.split(".")[:-1])
list_save_name='{File_Name}_radius_Fluorescence_distributions.hdf5'.format(File_Name = File_save_names)

with h5py.File(list_save_name, "w") as f:
      f.create_dataset('Radius_Fluo', data = r_fluo_dict)

'''

root = tk.Tk()
root.withdraw()

GUV_Post_Analysis_df_list = []

my_filetypes = [('all files', '.*'),('Image files', '.hdf5')]

filez = filedialog.askopenfilenames(parent = root, title='Please Select a File', filetypes = my_filetypes)

file_name= root.tk.splitlist(filez)[0]

f = h5py.File(file_name, 'r')

r_fluo= f['Radius_Fluo'][:]
#r_fluo= f['Radius_Fluo'][...].tolist()

#r_fluo_new = ast.literal_eval(r_fluo)
r_fluo_new = []
for item in r_fluo:
    if item [0] > 1:
        r_fluo_new.append(item)
    
    if item[0] < 1:
        if item[1] > 1200:
            r_fluo_new.append(item)

'''
# define a function to extract binding coefficient, return coefficient 
def coef_extractor(binding_dict,num):
    binding_list = binding_dict[num]
    radi = np.array([x[0] for x in binding_list]).reshape(-1, 1)
    binding = np.array([x[1] for x in binding_list])
    
    #fit linear regression model

    regr = linear_model.LinearRegression()
    regr.fit(radi, binding)

    coef = regr.coef_
    binding_y_pred = regr.predict(radi)

    plt.plot(radi,binding,'or',markersize=10)
    plt.plot(radi,binding_y_pred,'b-', linewidth=6)

    plt.show()

    return coef

# Get the list of coefficient
coef_list = []
for n in range(61):
   coef_n = coef_extractor(r_fluo_new,n)

   coef_list.append(coef_n)

plt.plot(coef_list,'ro-')
plt.show()
'''

radi = np.array([x[0] for x in r_fluo_new]).reshape(-1, 1)
print(radi.shape)
binding = np.array([x[1] for x in r_fluo_new])

print(radi)
print(binding)
regr = linear_model.LinearRegression()
regr.fit(radi, binding)

print(regr.coef_)
print(regr.intercept_)
binding_y_pred = regr.predict(radi)

fig, ax = plt.subplots()
fig.set_size_inches(12, 8)

ax.plot(radi,binding,'or',markersize=10)
ax.plot(radi,binding_y_pred,'b-', linewidth=6)


ax.set_xlabel('Radius (um)', fontsize = 16, fontweight = 'bold',color = 'k')   
ax.set_ylabel('Fluorescence Binding Intensities', fontsize = 16, fontweight = 'bold',color = 'K')

ax.tick_params(axis = 'y', labelcolor = 'k',labelsize = 'x-large')
ax.tick_params(axis = 'x',  labelsize = 'x-large',labelcolor = 'k')
ax.set_ylim(0,2400)
#plt.show()
fig_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving figure:",filetypes=[('Graph', '.tif')])
plt.savefig(fig_save_name, transparent=True)
