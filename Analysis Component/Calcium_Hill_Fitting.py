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
import seaborn
import matplotlib
import math
plt.rcParams["figure.figsize"] = (14,14)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#Ignore Warnings
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)

# Make a function to combine all df together into one df. Input is the directory of all dfs
def curve_df_combine(directory,name):
   os.chdir(directory)

   df_list = []

   df_filenames = glob.glob('*analysis.hdf5' )


   for n in range(len(df_filenames)):
      df_name = df_filenames[n]
      store = HDFStore(df_name)

      for key in store.keys():
         df = store[key]
         #df['GFP intensity'] = df['GFP intensity']
         df['Conc'] = name
         end_df = df.tail(1)
         df_list.append(end_df)

      store.close()


   df_len = len(df_list)

   df_final = pd.concat(df_list)
   df_final['conc'] = name
   return df_final, df_len
###################################################################################################
root = tk.Tk()
root.withdraw()
d = filedialog.askdirectory(initialdir=os.getcwd())
dirs = glob.glob(d + "/*/")

print(dirs)
df_fin = []
mean_conc_list = []
for direct in dirs:
   concentration = float(direct.split('/')[-2])

   df_final,_ = curve_df_combine(direct,concentration)

   df_fin.append(df_final)
   
   # Obtain standard error
   sem = df_final['Normalized GFP intensity'].sem()

   mean = df_final['Normalized GFP intensity'].mean() - sem

   mean_conc_list.append((mean,concentration))

mean_conc_list = sorted(mean_conc_list, key = lambda x: x[1])

df_fin = pd.concat(df_fin)
# Using map for 0 index
mean_list = map(lambda x: x[0], mean_conc_list)
mean_list = np.array(list(mean_list))
print(mean_list)
# Using map for 1 index
concentration_list = map(lambda x: x[1], mean_conc_list)
concentration_list = np.array(list(concentration_list))
# converting to list


#print(df_fin)

#isotherm_df = pd.DataFrame.from_dict({'Conc':concentration_list, 'Intensity': mean_list})

#print(isotherm_df)
background = mean_list[0].copy()
mean_list = mean_list - background
a = df_fin['Normalized GFP intensity']
df_fin['Normalized GFP intensity'] = df_fin['Normalized GFP intensity'] -background

#print (df_fin['Normalized GFP intensity']==a)
print(mean_list)
max_intensity = max(mean_list)

def func(x,max_int,ca_half,h):
   return max_int* (x**h /(ca_half **h + x**h))


#x_data = [0,100,200,350,500,800,1000]
#y_data = mean_list

#concentration_list = concentration_list[[0,4,5,6,7,8]]
#mean_list = mean_list[[0,4,5,6,7,8]]
popt, pcov = curve_fit(func,concentration_list,mean_list,p0=(max_intensity,20,1))

###############################################################################################

root = tk.Tk()
root.withdraw()
d1 = filedialog.askdirectory(initialdir=os.getcwd())
dirs = glob.glob(d1 + "/*/")

print(dirs)
df_fin1 = []
mean_conc_list1 = []
for direct in dirs:
   concentration = float(direct.split('/')[-2])

   df_final,_ = curve_df_combine(direct,concentration)

   df_fin1.append(df_final)

   mean = df_final['Normalized GFP intensity'].mean()

   mean_conc_list1.append((mean,concentration))

mean_conc_list1 = sorted(mean_conc_list1, key = lambda x: x[1])

df_fin1 = pd.concat(df_fin1)
# Using map for 0 index
mean_list1 = map(lambda x: x[0], mean_conc_list1)
mean_list1 = np.array(list(mean_list1))
print(mean_list1)
# Using map for 1 index
concentration_list1 = map(lambda x: x[1], mean_conc_list1)
concentration_list1 = np.array(list(concentration_list1))
# converting to list


#print(df_fin)

#isotherm_df = pd.DataFrame.from_dict({'Conc':concentration_list, 'Intensity': mean_list})

#print(isotherm_df)
background1 = mean_list1[0].copy()
mean_list1 = mean_list1 - background1

df_fin1['Normalized GFP intensity'] = df_fin1['Normalized GFP intensity']-background1

#print (df_fin['Normalized GFP intensity']==a)
print(mean_list1)
max_intensity1 = max(mean_list1)



#x_data = [0,100,200,350,500,800,1000]
#y_data = mean_list

#concentration_list = concentration_list[[0,4,5,6,7,8]]
#mean_list = mean_list[[0,4,5,6,7,8]]
popt1, pcov1 = curve_fit(func,concentration_list1,mean_list1,p0=(max_intensity1,20,1))

# Put two data frame together into 1 giant data frame
df_fin['condition'] = d.split('/')[-1]
df_fin1['condition'] = d1.split('/')[-1]
df_fin_total_list = [df_fin, df_fin1]
df_fin_total = pd.concat(df_fin_total_list)

'''
#################################################################################################
root = tk.Tk()
root.withdraw()
d = filedialog.askdirectory()
dirs = glob.glob(d + "/*/")

print(dirs)
df_fin2 = []
mean_conc_list2 = []
for direct in dirs:
   concentration = float(direct.split('\\')[1])

   df_final,_ = curve_df_combine(direct,concentration)

   df_fin2.append(df_final)

   mean = df_final['Normalized GFP intensity'].mean()

   mean_conc_list2.append((mean,concentration))

mean_conc_list2 = sorted(mean_conc_list2, key = lambda x: x[1])

df_fin2 = pd.concat(df_fin2)
# Using map for 0 index
mean_list2 = map(lambda x: x[0], mean_conc_list2)
mean_list2 = np.array(list(mean_list2))
print(mean_list2)
# Using map for 1 index
concentration_list2 = map(lambda x: x[1], mean_conc_list2)
concentration_list2 = np.array(list(concentration_list2))
# converting to list


#print(df_fin)

#isotherm_df = pd.DataFrame.from_dict({'Conc':concentration_list, 'Intensity': mean_list})

#print(isotherm_df)
background2 = mean_list2[0].copy()
mean_list2 = mean_list2 - background2
#a = df_fin['Normalized GFP intensity']
df_fin2['Normalized GFP intensity'] = df_fin2['Normalized GFP intensity']-background2

#print (df_fin['Normalized GFP intensity']==a)
print(mean_list2)
max_intensity2 = max(mean_list2)



#x_data = [0,100,200,350,500,800,1000]
#y_data = mean_list

#concentration_list = concentration_list[[0,4,5,6,7,8]]
#mean_list = mean_list[[0,4,5,6,7,8]]
popt2, pcov2 = curve_fit(func,concentration_list2,mean_list2,p0=(max_intensity2,20,1))
'''
############################################################################################

print(popt)
#print(popt1)
#print(popt2)

sns.set(style="ticks")
sns.set_context("paper", font_scale=1.5, rc={"font.size":20,"axes.labelsize":16, 'figure.figsize':(40,38),'lines.markersize': 14, "lines.linewidth": 4 })

fig, ax = plt.subplots()
fig.set_size_inches(12, 8)

ax = sns.lineplot(x='conc', y='Normalized GFP intensity', style='condition', markers=["o","^"],palette=['r','b'],hue ='condition', data=df_fin_total,err_style='bars',err_kws={'capsize':12, 'elinewidth':3,'capthick':4})
for l in ax.lines:
   l.set_linestyle("None")
#ax = sns.lineplot(x='conc', y='Normalized GFP intensity',marker='o', data=df_fin,err_style='bars')
#ax = sns.lineplot(x='conc', y='Normalized GFP intensity',marker='o', data=df_fin1,err_style='bars')

#ax = sns.lineplot(x='conc', y='Normalized GFP intensity',marker='o', data=df_fin2,err_style='bars')
ax.set_xscale('log')

#ax.lines[4].set_linestyle("None")
ax.plot(np.linspace(0,10000,400000), func(np.linspace(0,10000,400000), *popt),'r-')
ax.plot(np.linspace(0,10000,400000), func(np.linspace(0,10000,400000), *popt1),'b-')
#ax.plot(np.linspace(0,10000,400000), func(np.linspace(0,10000,400000), *popt2),'-')
ax.set_xlabel('Calcium Concentrations (um)', fontweight = 'bold', fontsize = 20)
ax.set_ylabel('cPla2 C2 Binding Intensities', fontweight = 'bold', fontsize = 20)
ax.legend(fontsize='large')
#ax.set_ylim(0,1600)
#ax.yaxis.set_ticks([0,130,260,390,520,650,780])
#ax.xaxis.set_ticklabels(concentration_list)
ax.set_ylim([-60,2700])
#ax.set_xlim([-60,2100])
#seaborn.despine(ax=ax, offset=0)
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.tight_layout()
#plt.show()

# Save plot as pdf for publication
fig_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving figure:",filetypes=[('Graph', '.pdf')])
plt.savefig(fig_save_name, transparent=True)
