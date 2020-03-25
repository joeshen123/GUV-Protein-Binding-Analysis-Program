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

import math
plt.rcParams["figure.figsize"] = (14,14)
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
d = filedialog.askdirectory()
dirs = glob.glob(d + "/*/")

print(dirs)
df_fin = []
mean_conc_list = []
for direct in dirs:
   concentration = float(direct.split('\\')[1])

   df_final,_ = curve_df_combine(direct,concentration)

   df_fin.append(df_final)

   mean = df_final['Normalized GFP intensity'].mean()

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
d = filedialog.askdirectory()
dirs = glob.glob(d + "/*/")

print(dirs)
df_fin1 = []
mean_conc_list1 = []
for direct in dirs:
   concentration = float(direct.split('\\')[1])

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
print(popt1)
#print(popt2)
sns.set(style="ticks")
sns.set_context("paper", font_scale=1.5, rc={"font.size":20,"axes.labelsize":16, 'figure.figsize':(40,38)})

ax = sns.lineplot(x='conc', y='Normalized GFP intensity',marker='o', data=df_fin,err_style='bars')
ax = sns.lineplot(x='conc', y='Normalized GFP intensity',marker='o', data=df_fin1,err_style='bars')
#ax = sns.lineplot(x='conc', y='Normalized GFP intensity',marker='o', data=df_fin2,err_style='bars')
ax.set_xscale('log')
ax.lines[0].set_linestyle("None")
ax.lines[2].set_linestyle("None")
#ax.lines[4].set_linestyle("None")
ax.plot(np.linspace(0,10000,400000), func(np.linspace(0,10000,400000), *popt),'-')
ax.plot(np.linspace(0,10000,400000), func(np.linspace(0,10000,400000), *popt1),'-')
#ax.plot(np.linspace(0,10000,400000), func(np.linspace(0,10000,400000), *popt2),'-')
ax.set_xlabel('Calcium Concentrations (um)', fontweight = 'bold', fontsize = 20)
ax.set_ylabel('Alox12 PLAT Binding Fluorescence Intensities', fontweight = 'bold', fontsize = 20)

#ax.set_ylim(0,1600)
ax.yaxis.set_ticks([0,130,260,390,520,650,780])
#ax.xaxis.set_ticklabels(concentration_list)

seaborn.despine(ax=ax, offset=0)
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.tight_layout()
plt.show()


#ch4_isotherm = pyiast.ModelIsotherm(isotherm_df,loading_key="Intensity",pressure_key="Conc", model="Langmuir")

#print(ch4_isotherm.params)

#pyiast.plot_isotherm(ch4_isotherm)



#print(df)
#sns.set(style="ticks")
#sns.set_context("paper", font_scale=1.5, rc={"font.size":12,"axes.labelsize":12, 'figure.figsize':(40,38)})
#ax= df_fin.plot.line(x='Conc', y='GFP intensity')
#ax.lines[0].set_linestyle("None")

#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_two,label='%s (n = %d)' %(label_2, df_final_two_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_three,label='%s (n = %d)' %(label_3, df_final_three_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_four,label='%s (n = %d)' %(label_4, df_final_four_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_five,label='%s (n = %d)' %(label_5, df_final_five_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_six,label='%s (n = %d)' %(label_6, df_final_six_len))
#ax.set_xlabel('Time Points (min)', fontweight = 'bold')
#ax.set_ylabel('Fluorescnence Intensities', fontweight = 'bold')
#ax.set_title("%s and %s Binding Profile" %(label_1, label_2))
#ax.legend(fontsize='small')

#plt.show()
#plt.savefig('C2_versus_mutant_rupture.tif', dpi=400)
'''
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
plt.tight_layout()
plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
'''

#fig_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving figure:",filetypes=[('Graph', '.png')])
#plt.savefig(fig_save_name, dpi=500)
