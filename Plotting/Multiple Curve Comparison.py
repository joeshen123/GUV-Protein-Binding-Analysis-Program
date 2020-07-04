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

# Save specific font that can be recognized by Adobe Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#Ignore Warnings
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)



# Make a function to combine all df together into one df. Input is the directory of all dfs
def curve_df_combine1(directory):
   os.chdir(directory)

   df_list = []

   df_filenames = glob.glob('*analysis.hdf5' )

   
   for n in range(len(df_filenames)):
      df_name = df_filenames[n]
      print(df_name)
      store = HDFStore(df_name)

      for key in store.keys():
         df = store[key]
         df_list.append(df)

      store.close()


   df_len = len(df_list)

   df_final = pd.concat(df_list)

   return df_final, df_len

# Make a function to combine all df together into one df. Input is the directory of all dfs
def curve_df_combine(directory):
   os.chdir(directory)

   df_list = []

   df_filenames = glob.glob('*analysis.hdf5' )

   
   for n in range(len(df_filenames)):
      df_name = df_filenames[n]
      print(df_name)
      store = HDFStore(df_name)

      for key in store.keys():
         df = store[key]
         #print(len(df))
         df = df[0:60]
         df['Time Point'] = np.linspace(0, 60, num = 60)
         #df['Normalized GFP intensity'] = df['Normalized GFP intensity'] 
         df_list.append(df)

      store.close()


   df_len = len(df_list)

   df_final = pd.concat(df_list)

   return df_final, df_len

root = tk.Tk()
root.withdraw()

root.directory = filedialog.askdirectory(initialdir=os.path.dirname(os.getcwd()))
Name0 = root.directory
df_final_zero,_ = curve_df_combine1(Name0)
normalization_factor = df_final_zero['Normalized GFP intensity'].median()


root.directory = filedialog.askdirectory(parent = root,initialdir=os.path.dirname(os.getcwd()))

Name1 = root.directory
label_1 = Name1.split("/")[-1]


df_final_one,df_final_one_len = curve_df_combine(Name1)
df_final_one['Normalized GFP intensity'] = df_final_one['Normalized GFP intensity']/normalization_factor
print(df_final_one)

root.directory = filedialog.askdirectory(parent = root,initialdir=os.path.dirname(os.getcwd()))
Name2 = root.directory
label_2 = Name2.split("/")[-1]
df_final_two,df_final_two_len = curve_df_combine(Name2)
df_final_two['Normalized GFP intensity'] = df_final_two['Normalized GFP intensity']/normalization_factor
print(df_final_two)


#by_row_index = df_final_two.groupby(df_final_two.index)
#df_means_swell = by_row_index.mean()

'''
root.directory = filedialog.askdirectory(parent = root,initialdir=os.path.dirname(os.getcwd()))
Name3 = root.directory
label_3 = Name3.split("/")[-1]
df_final_three,df_final_three_len = curve_df_combine(Name3)

df_final_three['Normalized GFP intensity'] = df_final_three['Normalized GFP intensity']/normalization_factor
#by_row_index = df_final_three.groupby(df_final_three.index)
#df_means_calcium = by_row_index.mean()

root.directory = filedialog.askdirectory(initialdir=os.path.dirname(os.getcwd()))
Name4 = root.directory
label_4 = Name4.split("/")[-1]
df_final_four,df_final_four_len = curve_df_combine(Name4)
df_final_four['Normalized GFP intensity'] = df_final_four['Normalized GFP intensity']/normalization_factor
'''
'''
root.directory = filedialog.askdirectory(initialdir=os.path.dirname(os.getcwd()))
Name5 = root.directory
label_5 = Name5.split("/")[-1]
df_final_five,df_final_five_len = curve_df_combine(Name5)
df_final_five['Normalized GFP intensity'] = df_final_five['Normalized GFP intensity']/normalization_factor
'''
'''
root.directory = filedialog.askdirectory()
Name6 = root.directory
label_6 = Name3.split("/")[-1]
df_final_six,df_final_six_len = curve_df_combine(Name6)
'''
'''
# Obtain the last measurement of protein binding on liposome. Then use 2-sampled ttest to determine whether these two afre significantly different
GFP_1 = df_final_one[df_final_one['Time Point']== 60.0]['Normalized GFP intensity'].to_list()
GFP_2 = df_final_two[df_final_two['Time Point']== 60.0]['Normalized GFP intensity'].to_list()

print(scipy.stats.ttest_ind(GFP_1, GFP_2)) 
'''

#print(df)

sns.set_context("paper", font_scale=1.5, rc={"font.size":12,"axes.labelsize":12,"lines.linewidth": 5,'lines.markersize': 7})
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)
'''
for a in df_final_one:
    ax.plot(a['Time Point'],a['Normalized GFP intensity'])

plt.show()
'''

#ax= df_fin.plot.line(x='Conc', y='GFP intensity')
#ax.lines[0].set_linestyle("None")

ax = sns.lineplot(x='Time Point', y='Normalized GFP intensity', data = df_final_one, label='%s (n = %d)' %(label_1, df_final_one_len))


ax = sns.lineplot(x='Time Point', y='Normalized GFP intensity', data = df_final_two,label='%s (n = %d)' %(label_2, df_final_two_len))
#ax = sns.lineplot(x='Time Point', y='Normalized GFP intensity', data = df_final_three,label='%s (n = %d)' %(label_3, df_final_three_len))
#ax = sns.lineplot(x='Time Point', y='Normalized GFP intensity', data = df_final_four,label='%s (n = %d)' %(label_4, df_final_four_len))
#ax = sns.lineplot(x='Time Point', y='Normalized GFP intensity', data = df_final_five,label='%s (n = %d)' %(label_5, df_final_five_len))


ax.tick_params(axis = 'y', labelsize = 'x-large')
ax.tick_params(axis = 'x',  labelsize = 'x-large')
ax.set_xlabel('Time Points (min)', fontweight = 'bold', fontsize = 20)
ax.set_ylabel('Normalized Alox12 PLAT Binding Intensities', fontweight = 'bold', fontsize = 20)
ax.set_ylim([0,24])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()
#plt.savefig('C2_versus_mutant_rupture.tif', dpi=400)


#fig_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving figure:",filetypes=[('Graph', '.pdf')])
#ass = AnySizeSaver(fig=fig, dpi=600, filename=fig_save_name)
#plt.show()

#plt.savefig(fig_save_name, transparent=True)

'''
File_save_names = filedialog.asksaveasfilename(parent=root,title="Please select a file name for saving:",filetypes=[('Image Files', '.hdf5')])
decay_Name='{File_Name}.hdf5'.format(File_Name = File_save_names)

with h5py.File(decay_Name, "w") as f:
      f.create_dataset('control', data = df_means_control, compression = 'gzip')
      f.create_dataset('swell', data = df_means_swell, compression = 'gzip')
      f.create_dataset('calcium', data = df_means_calcium, compression = 'gzip')
'''
