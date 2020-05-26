import matplotlib.pyplot as plt
import pickle
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from pandas import HDFStore
from scipy.interpolate import interp1d

import matplotlib

# Save specific font that can be recognized by Adobe Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class AnySizeSaver():
    def __init__(self, fig=None, figsize=None, dpi=None, filename=None):
        if not fig: fig=plt.gcf()
        self.fig = fig
        if not figsize: figsize=self.fig.get_size_inches()
        self.figsize=figsize
        if not dpi: dpi=self.fig.dpi
        self.dpi=dpi
        if not filename: filename="myplot.png"
        self.filename=filename
        self.cid = self.fig.canvas.mpl_connect("key_press_event", self.key_press)

    def key_press(self, event):
        if event.key == "t":
            self.save()

    def save(self):
        oldfigsize = self.fig.get_size_inches()
        olddpi=self.fig.dpi
        self.fig.set_size_inches(self.figsize)
        self.fig.set_dpi(self.dpi)
        self.fig.savefig(self.filename, dpi=self.dpi)
        self.fig.set_size_inches(oldfigsize, forward=True)
        self.fig.set_dpi(olddpi)
        self.fig.canvas.draw_idle()
        print(fig.get_size_inches())



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

Time = np.linspace(0, 15, num = len(Time))

print(Time)

radius = df['radius_micron'].values
#volume = df['volume_micron_cube'].values
#volume = df['volume_micron_cube'].values
#volume = df['Cell Volume'].values
GFP_Intensity = df['Normalized GFP intensity']/1000
GFP_Intensity = GFP_Intensity.values
'''
# Find the indices of max bindings and subset the data only include decay points
max_ind = np.argmax(GFP_Intensity)

decay_intensity = GFP_Intensity[max_ind:]
decay_Time = Time[max_ind:]
decay_radius = radius[max_ind:]
'''
#Convert decay _Time
'''
for n in range(len(decay_Time)):
    decay_Time[n] = n * 10
'''
fig, ax1 = plt.subplots()
fig.set_size_inches(12, 8)
ax1.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold',color = 'k')
#ax1.set_ylabel('Cpla2 C2 Bindings', fontsize = 16,fontweight = 'bold',color ='r')
ax1.set_ylabel('GUV Radius (um)', fontsize = 16,fontweight = 'bold',color = 'k')

plt1 = ax1.plot(Time, radius,'ko-', markersize=12, linewidth=4)
#plt1 = ax1.plot(Time, radius, 'b-')

ax1.tick_params(axis = 'y', labelcolor = 'k',labelsize = 'x-large')
ax1.tick_params(axis = 'x',  labelsize = 'x-large',labelcolor = 'k')

ax2 = ax1.twinx()
#ax2.set_ylim((1.2,3.2))
ax2.set_xlabel('Time Points (sec)', fontsize = 16, fontweight = 'bold',color = 'k')
ax2.set_ylabel('cPla2 C2 Binding Intensities', fontsize = 16,fontweight = 'bold',color='r')
plt2 = ax2.plot(Time, GFP_Intensity, 'ro-', markersize=12,linewidth=4)
ax2.tick_params(axis = 'y', labelcolor = 'r',labelsize = 'x-large')

'''
# Fitting with exponential decay
def func(x, a, b,c):
   return a * np.exp(-b*x)+c 

popt, pcov = curve_fit(func, decay_Time,decay_intensity,p0=(1,1e-7,1))
popt1, pcov1 = curve_fit(func, decay_Time,decay_radius,p0=(1,1e-10,1))
# Plot the fitted curve
ax1.plot(np.linspace(decay_Time[0],decay_Time[-1],400000), func(np.linspace(decay_Time[0],decay_Time[-1],400000), *popt1),'k--',linewidth=6)
ax2.plot(np.linspace(decay_Time[0],decay_Time[-1],400000), func(np.linspace(decay_Time[0],decay_Time[-1],400000), *popt),'b--',linewidth=6)
'''
#print(popt)
#print(popt1)

plt_total = plt2 + plt1

fig.tight_layout()
#plt.show()

fig_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving figure:",filetypes=[('Graph', '.pdf')])

plt.savefig(fig_save_name, transparent=True)

#ass = AnySizeSaver(fig=fig, dpi=600, filename=fig_save_name)
#plt.show()
#plt.savefig(fig_save_name, dpi=300)

