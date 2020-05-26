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
from scipy.stats import ttest_ind
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
   df_final['name'] = name

   return df_final, df_len



root = tk.Tk()
root.withdraw()
d = filedialog.askdirectory()



df_final,_ = curve_df_combine(d,'Hypotonic Rupture')






root = tk.Tk()
root.withdraw()
d1 = filedialog.askdirectory()


df_final1,_ = curve_df_combine(d1,'Triton x-100 Rupture')



df_fin1 = [df_final,df_final1]

df_fin1= pd.concat(df_fin1)

print(df_fin1)

hypo = df_fin1[df_fin1['name'] == 'Hypotonic Rupture']
triton = df_fin1[df_fin1['name'] == 'Triton x-100 Rupture']

p_value = ttest_ind(hypo['Normalized GFP intensity'],triton['Normalized GFP intensity'])

print (p_value)

fig,ax = plt.subplots()

#print(df)
sns.set(style="ticks")
sns.set_context("paper", font_scale=1.5, rc={"font.size":16,"axes.labelsize":12, 'figure.figsize':(40,38)})

ax = sns.barplot(x="name", y="Normalized GFP intensity", data=df_fin1)
ax = sns.swarmplot(x="name", y="Normalized GFP intensity", data=df_fin1, color='.2', size = 8)
#ax= df_fin.plot.line(x='Conc', y='GFP intensity')
#ax.lines[0].set_linestyle("None")

#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_two,label='%s (n = %d)' %(label_2, df_final_two_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_three,label='%s (n = %d)' %(label_3, df_final_three_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_four,label='%s (n = %d)' %(label_4, df_final_four_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_five,label='%s (n = %d)' %(label_5, df_final_five_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_six,label='%s (n = %d)' %(label_6, df_final_six_len))
#ax.set_xlabel('cPla2 C2 concentration (nm)', fontweight = 'bold')
ax.set_ylabel('Alox5 PLAT binding Intensities', fontweight = 'bold')
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
