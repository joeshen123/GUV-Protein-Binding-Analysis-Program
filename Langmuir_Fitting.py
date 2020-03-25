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
         #df['GFP intensity'] = df['GFP intensity']
         end_df = df.tail(1)
         df_list.append(end_df)

      store.close()


   df_len = len(df_list)

   df_final = pd.concat(df_list)
   df_final['conc'] = name

   return df_final, df_len

##########################################################################
root = tk.Tk()
root.withdraw()
d = filedialog.askdirectory()
dirs = glob.glob(d + "/*/")

concentration_list = []
df_fin = []
mean_list = []
for direct in dirs:
   concentration = float(direct.split('\\')[1])
   concentration_list.append(concentration)

   df_final,_ = curve_df_combine(direct,concentration)


   df_fin.append(df_final)

   mean = df_final['Normalized GFP intensity'].mean()

   mean_list.append(mean)

df_fin= pd.concat(df_fin)
print(concentration_list)
print(mean_list)
#####################################################################################

root = tk.Tk()
root.withdraw()
d1 = filedialog.askdirectory()
dirs1 = glob.glob(d1 + "/*/")

concentration_list1 = []
df_fin1 = []
mean_list1 = []
for direct in dirs1:
   concentration = float(direct.split('\\')[1])
   concentration_list1.append(concentration)

   df_final,_ = curve_df_combine(direct,concentration)

   df_fin1.append(df_final)

   mean = df_final['Normalized GFP intensity'].mean()

   mean_list1.append(mean)

df_fin1= pd.concat(df_fin1)
print(concentration_list1)
print(mean_list1)


#############################################################################
root = tk.Tk()
root.withdraw()
d2 = filedialog.askdirectory()
dirs2 = glob.glob(d2 + "/*/")

concentration_list2 = []
df_fin2 = []
mean_list2 = []
for direct in dirs2:
   concentration = float(direct.split('\\')[1])
   concentration_list2.append(concentration)

   df_final,_ = curve_df_combine(direct,concentration)

   df_fin2.append(df_final)

   mean = df_final['Normalized GFP intensity'].mean()

   mean_list2.append(mean)

df_fin2= pd.concat(df_fin2)
print(concentration_list2)
print(mean_list2)

#######################################################################################
#isotherm_df = pd.DataFrame.from_dict({'Conc':concentration_list, 'Intensity': mean_list})

#print(isotherm_df)

#mean_list = mean_list - mean_list[0]

# Calculate the affinity and other parameters
def func(x,Imax,Kd):
   return Imax / (1 + (Kd / x))


#x_data = [0,100,200,350,500,800,1000]
#y_data = mean_list

popt, pcov = curve_fit(func, concentration_list,mean_list,p0=(max(mean_list),1))
popt1, pcov1 = curve_fit(func, concentration_list1,mean_list1,p0=(max(mean_list1),1))
popt2, pcov2 = curve_fit(func, concentration_list2,mean_list2,p0=(max(mean_list2),1))

print(popt)
print(popt1)
print(popt2)

# Plot the graph
sns.set(style="ticks")
sns.set_context("paper", font_scale=1.5, rc={"font.size":16,"axes.labelsize":12, 'figure.figsize':(40,38)})

ax = sns.lineplot(x='conc', y='Normalized GFP intensity',marker='o', data=df_fin,err_style='bars')
ax.lines[0].set_linestyle("None")
ax.plot(np.linspace(0,2000,40000), func(np.linspace(0,2000,40000), *popt),'-')
ax = sns.lineplot(x='conc', y='Normalized GFP intensity',marker='o', data=df_fin1,err_style='bars')
ax.lines[3].set_linestyle("None")
ax.plot(np.linspace(0,2000,40000), func(np.linspace(0,2000,40000), *popt1),'-')
ax = sns.lineplot(x='conc', y='Normalized GFP intensity',marker='o', data=df_fin2,err_style='bars')
ax.lines[6].set_linestyle("None")
ax.plot(np.linspace(0,2000,40000), func(np.linspace(0,2000,40000), *popt2),'-')

#ax= df_fin.plot.line(x='Conc', y='GFP intensity')
#ax.lines[0].set_linestyle("None")

#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_two,label='%s (n = %d)' %(label_2, df_final_two_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_three,label='%s (n = %d)' %(label_3, df_final_three_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_four,label='%s (n = %d)' %(label_4, df_final_four_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_five,label='%s (n = %d)' %(label_5, df_final_five_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_six,label='%s (n = %d)' %(label_6, df_final_six_len))
ax.set_xlabel('Protein Concentration (nm)', fontweight = 'bold')
ax.set_ylabel('Protein Binding Intensities', fontweight = 'bold')
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
