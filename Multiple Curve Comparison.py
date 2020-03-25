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

def smooth (y, box_pts=3):
  box = np.ones(box_pts) / box_pts
  y_smooth = np.convolve(y, box, 'same')

  return y_smooth

#Ignore Warnings
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)


def Extract_df_mean(directory):
    os.chdir(directory)

    df_list = []

    df_filenames = glob.glob('*analysis.hdf5' )



    for n in range(len(df_filenames)):
       df_name = df_filenames[n]
       store = HDFStore(df_name)

       for key in store.keys():
          df = store[key]

          df_list.append(df)

       store.close()

    df_final = pd.concat(df_list)

    intensity_mean = df_final['Normalized GFP intensity'].mean()

    return intensity_mean


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

   df_final = pd.concat(df_list)

   return df_final, df_len

root = tk.Tk()
root.withdraw()


root.directory = filedialog.askdirectory()

Name1 = root.directory
label_1 = Name1.split("/")[-1]


df_final_one,df_final_one_len = curve_df_combine(Name1)

print(df_final_one)

#by_row_index = df_final_one.groupby(df_final_one.index)
#df_means_control = by_row_index.mean()
#root.directory = filedialog.askdirectory()

#df_mean2 = Extract_df_mean(root.directory)


root.directory = filedialog.askdirectory()
Name2 = root.directory
label_2 = Name2.split("/")[-1]
df_final_two,df_final_two_len = curve_df_combine(Name2)

#by_row_index = df_final_two.groupby(df_final_two.index)
#df_means_swell = by_row_index.mean()


root.directory = filedialog.askdirectory()
Name3 = root.directory
label_3 = Name3.split("/")[-1]
df_final_three,df_final_three_len = curve_df_combine(Name3)

'''
#by_row_index = df_final_three.groupby(df_final_three.index)
#df_means_calcium = by_row_index.mean()

root.directory = filedialog.askdirectory()
Name4 = root.directory
label_4 = Name4.split("/")[-1]
df_final_four,df_final_four_len = curve_df_combine(Name4)


root.directory = filedialog.askdirectory()
Name5 = root.directory
label_5 = Name5.split("/")[-1]
df_final_five,df_final_five_len = curve_df_combine(Name5)
'''
'''
root.directory = filedialog.askdirectory()
Name6 = root.directory
label_6 = Name3.split("/")[-1]
df_final_six,df_final_six_len = curve_df_combine(Name6)
'''


#print(df)
sns.set(style="ticks")
sns.set_context("paper", font_scale=1.5, rc={"font.size":12,"axes.labelsize":12})
fig, ax = plt.subplots()
fig.set_size_inches(12, 6)
#ax= df_fin.plot.line(x='Conc', y='GFP intensity')
#ax.lines[0].set_linestyle("None")

ax = sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_one,label='%s (n = %d)' %(label_1, df_final_one_len))



#ax.legend().set_visible(False)
#plt.draw()

#ax2 = plt.twinx()

ax = sns.lineplot(x='Time Point', y='GFP intensity',data = df_final_two,label='%s (n = %d)' %(label_2, df_final_two_len))

#ax2.tick_params(axis = 'y', labelcolor = 'b', labelsize = 'x-large')
#ax2.tick_params(axis = 'x',  labelsize = 'x-large')
#ax2.set_xlabel('Time Points (min)', fontweight = 'bold', fontsize = 20)
#ax2.set_ylabel('Normalize Fluorescence Intensities', fontweight = 'bold', fontsize = 20, color = 'b')

ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_three,label='%s (n = %d)' %(label_3, df_final_three_len))
#ax= sns.lineplot(x='Time Point', y='Normalized GFP intensity', data = df_final_four,label='%s (n = %d)' %(label_4, df_final_four_len))
#ax= sns.lineplot(x='Time Point', y='Normalized GFP intensity', data = df_final_five,label='%s (n = %d)' %(label_5, df_final_five_len))

#lines, labels = ax.get_legend_handles_labels()
#lines2, labels2 = ax2.get_legend_handles_labels()
#ax2.legend(lines + lines2, labels + labels2, loc=0)

#ax2.set_ylabel('Normalize Fluorescence Intensities', fontweight = 'bold', fontsize = 20)
#ax2.set_xlabel('Time Points (min)', fontweight = 'bold', fontsize = 20)
#ax.set_title("%s and %s Binding Profile" %(label_1, label_2))
#ax.legend(fontsize='small')
#ax2.legend(fontsize='small')

#ax.set_ylim([2,12])
#ax2.set_ylim([0,10])

ax.tick_params(axis = 'y', labelsize = 'x-large')
ax.tick_params(axis = 'x',  labelsize = 'x-large')
ax.set_xlabel('Time Points (min)', fontweight = 'bold', fontsize = 20)
ax.set_ylabel('Fluorescence Intensities', fontweight = 'bold', fontsize = 16)
#ax.set_ylim([0.1,1.4])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.tight_layout()
#plt.show()
#plt.savefig('C2_versus_mutant_rupture.tif', dpi=400)

#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#plt.tight_layout()
#plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.get_legend().remove()
#plt.tight_layout()
plt.show()

#fig_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving figure:",filetypes=[('Graph', '.png')])
#plt.savefig(fig_save_name, dpi=1000)

'''
File_save_names = filedialog.asksaveasfilename(parent=root,title="Please select a file name for saving:",filetypes=[('Image Files', '.hdf5')])
decay_Name='{File_Name}.hdf5'.format(File_Name = File_save_names)

with h5py.File(decay_Name, "w") as f:
      f.create_dataset('control', data = df_means_control, compression = 'gzip')
      f.create_dataset('swell', data = df_means_swell, compression = 'gzip')
      f.create_dataset('calcium', data = df_means_calcium, compression = 'gzip')
'''
