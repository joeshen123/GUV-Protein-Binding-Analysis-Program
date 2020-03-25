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
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import matplotlib as mpl
#mpl.rcParams['savefig.dpi'] = 1000
plt.rcParams["figure.figsize"] = (14,14)
#Ignore Warnings
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)

# Make a function to combine all df together into one df. Input is the directory of all dfs
def curve_df_combine(directory,name):
   os.chdir(directory)

   df_list = []

   df_filenames = glob.glob('*.hdf5' )


   for n in range(len(df_filenames)):
      df_name = df_filenames[n]
      store = HDFStore(df_name)

      for key in store.keys():
         df = store[key]
         end_df = df.tail(1)
         df_list.append(end_df)

      store.close()


   df_len = len(df_list)

   df_final = pd.concat(df_list)
   df_final['conc'] = int(name)

   return df_final, df_len

def curve_df_combine2(directory,name,total):
   os.chdir(directory)

   df_list = []

   df_filenames = glob.glob('*.hdf5' )


   for n in range(len(df_filenames)):
      df_name = df_filenames[n]
      store = HDFStore(df_name)

      for key in store.keys():
         df = store[key]
         end_df = df.tail(1)
         df_list.append(end_df)

      store.close()


   df_len = len(df_list)

   df_final = pd.concat(df_list)
   df_final['conc'] = int(name)
   df_final['Normalized GFP intensity'] = df_final['Normalized GFP intensity'] /total

   return df_final, df_len

root = tk.Tk()
root.withdraw()
d = filedialog.askdirectory()
dirs = glob.glob(d + "/*/")



mean_dict = {}
for direct in dirs:
   concentration = float(direct.split('\\')[1])


   df_final,_ = curve_df_combine(direct,concentration)


   mean = df_final['Normalized GFP intensity'].mean()

   mean_dict[concentration] = mean



root = tk.Tk()
root.withdraw()
d1 = filedialog.askdirectory()
dirs1 = glob.glob(d1 + "/*/")


df_fin1 = []

for direct in dirs1:
   concentration = float(direct.split('\\')[1])


   df_final,_ = curve_df_combine2(direct,concentration,mean_dict[concentration])

   df_fin1.append(df_final)



df_fin1= pd.concat(df_fin1)

print(df_fin1)



fig,ax = plt.subplots()

#print(df)
sns.set(style="ticks")
sns.set_context("paper", font_scale=1.5, rc={"font.size":16,"axes.labelsize":12, 'figure.figsize':(40,38)})

ax = sns.barplot(x="conc", y="Normalized GFP intensity", data=df_fin1)
ax = sns.swarmplot(x="conc", y="Normalized GFP intensity", data=df_fin1, color='.2', size = 8)
#ax= df_fin.plot.line(x='Conc', y='GFP intensity')
#ax.lines[0].set_linestyle("None")

#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_two,label='%s (n = %d)' %(label_2, df_final_two_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_three,label='%s (n = %d)' %(label_3, df_final_three_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_four,label='%s (n = %d)' %(label_4, df_final_four_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_five,label='%s (n = %d)' %(label_5, df_final_five_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_six,label='%s (n = %d)' %(label_6, df_final_six_len))
ax.set_xlabel('cPla2 C2 concentration (nm)', fontweight = 'bold')
ax.set_ylabel('cPla2 C2 binding intensity ratio', fontweight = 'bold')
##ax.set_title("%s and %s Binding Profile" %(label_1, label_2))
#ax.legend(fontsize='small')
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.tight_layout()
plt.show()
#plt.savefig('C2_versus_mutant_rupture.tif', dpi=400)
'''
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
plt.tight_layout()
plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
'''

#fig_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving figure:",filetypes=[('Graph', '.png')])
#plt.savefig(fig_save_name, dpi=500)
