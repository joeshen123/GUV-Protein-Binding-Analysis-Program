import matplotlib.pyplot as plt
import pickle
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from pandas import HDFStore
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind
import scipy.stats as stats
import seaborn as sns
import matplotlib

# Save specific font that can be recognized by Adobe Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('Image files', '.hdf5')]

Original_test_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

my_filetypes = [('all files', '.*'),('Image files', '.hdf5')]

Original_test_path2 = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)



# Make a function to convert from hdf5 to pandas dataframe
def df_converter (path):
  store = HDFStore(path)

  df_list = []
  for key in store.keys():
     df = store[key].iloc[-1,:]
     df['Normalized GFP intensity'] = df['Normalized GFP intensity'] / 1000
     df_list.append(df)

  store.close()
  df_final = pd.concat(df_list, axis = 1).T

  return (df_final)

# Obtain FLIM data for Hypo and Iso
Hypo_Data = df_converter(Original_test_path)
Iso_Data = df_converter(Original_test_path2)

#Combine Iso and Hypo data into 1 dataframe
Hypo_Data['condition'] = 'Hypotonic'
Iso_Data['condition'] = 'Isotonic'
#df_fin2['condition'] = d2.split('/')[-1]
Hypo_Iso_list = [Iso_Data,Hypo_Data]
Hypo_Iso_Final = pd.concat(Hypo_Iso_list)


print(Hypo_Iso_Final)
# Perform ttest to determine significance
import scipy.stats
HYPO = Hypo_Iso_Final[Hypo_Iso_Final['condition']== 'Hypotonic']['Normalized GFP intensity'].to_list()
ISO = Hypo_Iso_Final[Hypo_Iso_Final['condition']== 'Isotonic']['Normalized GFP intensity'].to_list()

print(scipy.stats.ttest_ind(HYPO, ISO, equal_var=False)) 

# Drawthe point plot with individual data points

fig, ax = plt.subplots()
fig.set_size_inches(12, 8)

sns.set(style="ticks")
sns.set_context("paper", font_scale=2, rc={"font.size":16,"axes.labelsize":16})


ax = sns.pointplot(x="condition", y="Normalized GFP intensity", data=Hypo_Iso_Final, ci=95, color='k',capsize = 0.1, errwidth=4)
ax = sns.swarmplot(x="condition", y="Normalized GFP intensity",data=Hypo_Iso_Final, hue = 'condition',size = 8,palette=['b','r'])


#labels = ['Hypotonic \n(n = %d)' % len(Hypo),'Isotonic \n(n = %d)' % len(Iso)]

#fig.set(xticklabels = labels)

ax.tick_params(labelsize= 16)
plt.xlabel('')
plt.ylabel('Fluorescence Life Time (ns)', fontsize=14, fontweight = 'bold')
ax.set_ylim((0.8,3.2))
plt.rcParams['axes.labelweight'] = 'bold'
plt.show()

# Save as PDF
#fig_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving figure:",filetypes=[('Graph', '.pdf')])
#plt.savefig(fig_save_name, transparent=True)
