import matplotlib.pyplot as plt
import pickle
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from pandas import HDFStore
from scipy.interpolate import interp1d


root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('Image files', '.hdf5')]

Original_test_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

#root.destroy()
store = HDFStore(Original_test_path)

df_list = []
for key in store.keys():
    print(key)
    df = store[key]
    df_list.append(df)

    store.close()


df = df_list[0]

print(df)

Time = df['Time Point'].values

Time = np.linspace(0, 35, num = len(Time))

print(Time)

#radius = df['radius_micron'].values
volume = df['volume_micron_cube'].values
#volume = df['Cell Volume'].values
#GFP_Intensity = df['Normalized GFP intensity'].values
'''
# Find the indices of max bindings and subset the data only include decay points
max_ind = np.argmax(GFP_Intensity)

decay_intensity = GFP_Intensity[max_ind:]
decay_Time = Time[max_ind:]
decay_radius = radius[max_ind:]
#Convert decay _Time

for n in range(len(decay_Time)):
    decay_Time[n] = n * 10
'''
fig, ax1 = plt.subplots()

ax1.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold',color = 'k')
#ax1.set_ylabel('Cpla2 C2 Bindings', fontsize = 16,fontweight = 'bold',color ='r')
ax1.set_ylabel('GUV Volume (um^3)', fontsize = 16,fontweight = 'bold',color = 'k')

plt1 = ax1.plot(Time, volume,'bo-', markersize=10, linewidth=6)
#plt1 = ax1.plot(Time, radius, 'b-')

ax1.tick_params(axis = 'y', labelcolor = 'k',labelsize = 'x-large')
ax1.tick_params(axis = 'x',  labelsize = 'x-large',labelcolor = 'k')
'''
ax2 = ax1.twinx()
#x2.set_ylim((80,150))
ax2.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold')
ax2.set_ylabel('Alox12 PLAT Binding Intensities', fontsize = 16,fontweight = 'bold',color='r')
plt2 = ax2.plot(decay_Time, decay_intensity, 'ro', markersize=10)
ax2.tick_params(axis = 'y', labelcolor = 'r',labelsize = 'x-large')

# Fitting with exponential decay
def func(x, a, b,c):
   return a * np.exp(-b*x)+c 

popt, pcov = curve_fit(func, decay_Time,decay_intensity,p0=(1,1e-7,1))

# Plot the fitted curve
ax2.plot(np.linspace(decay_Time[0],decay_Time[-1],400000), func(np.linspace(decay_Time[0],decay_Time[-1],400000), *popt),'b--',linewidth=6)
print(popt)

plt_total = plt2 + plt1
'''
fig.tight_layout()
#plt.show()

fig_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving figure:",filetypes=[('Graph', '.tiff')])
plt.savefig(fig_save_name, dpi=300)

