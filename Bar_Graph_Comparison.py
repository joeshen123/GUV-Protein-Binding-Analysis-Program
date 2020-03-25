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
root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('Image files', '.hdf5')]

Original_test_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

df = pd.read_csv(Original_test_path)
df.drop('Unnamed: 3', axis = 1, inplace=True)
df.drop('Unnamed: 4', axis = 1, inplace=True)
df.drop('Unnamed: 0', axis = 1, inplace=True)
#print(df)

# Change the df to seaborn format
df = df.stack().reset_index()
df = df.drop(df.columns[0], axis=1)

df.columns = ['Tonicity','Life Time']

#print(df)

Hypo = df[df['Tonicity'] == 'Hypotonic']
Iso = df[df['Tonicity'] == 'Isotonic']

# Check whether two data are normal distributed
print(stats.normaltest(Hypo['Life Time']))
print(stats.normaltest(Iso['Life Time']))

# Check Ttest and determine whether they are significant
Hypo_Iso = ttest_ind(Hypo['Life Time'],Iso['Life Time'])

print(Hypo_Iso)
# Show the barplot
sns.set(style="ticks")
sns.set_context("paper", font_scale=1.5, rc={"font.size":12,"axes.labelsize":12})

sns.barplot(x="Tonicity", y="Life Time", data=df, capsize= .05, ci="sd")
fig = sns.swarmplot(x="Tonicity", y="Life Time", data=df, color='.2', size = 8)


labels = ['Hypotonic \n(n = %d)' % len(Hypo),'Isotonic \n(n = %d)' % len(Iso)]

fig.set(xticklabels = labels)

fig.tick_params(labelsize= 16)
plt.xlabel('')
plt.ylabel('Fluorescence Life Time (ns)', fontsize=14, fontweight = 'bold')
plt.ylim((4.82,5.18))
plt.rcParams['axes.labelweight'] = 'bold'
#plt.show()

graph_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving datasheet:",filetypes=[('Graph', '.png')])
plt.savefig(graph_name, dpi=1000)
