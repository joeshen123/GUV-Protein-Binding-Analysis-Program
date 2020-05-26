import matplotlib.pyplot as plt
import h5py
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.optimize import curve_fit

root = tk.Tk()
root.withdraw()


my_filetypes = [('all files', '.*'),('Image files', '.pkl')]

Original_test_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

f = h5py.File(Original_test_path, 'r')

Control = f['control'][:]

swell = f['swell'][:]

calcium = f['calcium'][:]
print(swell[:,0].shape )
time_point = calcium[:,0] * 60
control_data = Control[:,5]
swell_data = swell[:,5]
print(swell_data.shape)
calcium_data = calcium[:,5]


#df = df_list_one[0]
#list_num = df_list_one[0]['GFP intensity']
#df.plot(x = 'Time Point',y='GFP intensity',kind = 'scatter')
plt.plot(time_point,calcium_data,'bo')


#x_data = df_list_one[0].iloc[np.argmax(list_num):]['Time Point'].values
#print(x_data)
#y_data = df_list_one[0].iloc[np.argmax(list_num):]['GFP intensity'].values
#print(y_data)


def func(x, a, b, c):
   return a * np.exp(-b*x) + c

popt, pcov = curve_fit(func, time_point,calcium_data,p0=(1,1e-8,1))

print(popt)
#print(np.exp(-x_data))

plt.plot(time_point, func(time_point, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.legend(loc='best')
plt.show()

