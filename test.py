'''
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
import h5py

from tkinter import filedialog
root = tk.Tk()
root.withdraw()

dirs = []
while True:
    d = filedialog.askdirectory()
    if not d: break
    dirs.append(d)
    print (dirs)

my_filetypes = [('all files', '.*'),('Image files', '.hdf5')]

def Extract_df_mean(file_name):
    
    store = HDFStore(file_name)
    
    df_list = []
    for key in store.keys():
        df = store[key]

        df_list.append(df)
      
    store.close()
   
    df_final = pd.concat(df_list)

    intensity_mean = df_final['GFP intensity'].median()
   
    return intensity_mean

root = tk.Tk()
root.withdraw()

filez = filedialog.askopenfilename(parent = root, title='Please Select a File', filetypes = my_filetypes)

df_mean = Extract_df_mean(filez)

print(df_mean)

from skimage import data
from skimage import filters
from skimage import segmentation
from skimage import morphology
import time
import napari
coins = data.coins()

with napari.gui_qt():
   viewer = napari.view_image(coins, name='coins')

   edges = filters.sobel(coins)

   edges_layer = viewer.add_image(edges, colormap='magenta', blending='additive')


   pts_layer = viewer.add_points(size=5)
   pts_layer.mode = 'add'

   print(pts_layer)
   print("This is how you pause")

   input()

# annotate the background and all the coins, in that order
   coordinates = pts_layer.data
   coordinates_int = np.round(coordinates).astype(int)

   markers_raw = np.zeros_like(coins)
   markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(coordinates))
# raw markers might be in a little watershed "well".
   markers = morphology.dilation(markers_raw, morphology.disk(5))

   segments = segmentation.watershed(edges, markers=markers)

   labels_layer = viewer.add_labels(segments - 1)  # make background 0


Display one shapes layer ontop of one image layer using the add_shapes and
add_image APIs. When the window is closed it will print the coordinates of
your shapes.


import numpy as np
from skimage import data
import napari


with napari.gui_qt():
    # create the viewer and window
    viewer = napari.Viewer()

    # add the image
    photographer = data.camera()
    image_layer = napari.view_image(photographer, name='photographer')

    # create a list of polygons
    polygons = [
        np.array([[11, 13], [111, 113], [22, 246]]),
        np.array(
            [
                [505, 60],
                [402, 71],
                [383, 42],
                [251, 95],
                [212, 59],
                [131, 137],
                [126, 187],
                [191, 204],
                [171, 248],
                [211, 260],
                [273, 243],
                [264, 225],
                [430, 173],
                [512, 160],
            ]
        ),
        np.array(
            [
                [310, 382],
                [229, 381],
                [209, 401],
                [221, 411],
                [258, 411],
                [300, 412],
                [306, 435],
                [268, 434],
                [265, 454],
                [298, 461],
                [307, 461],
                [307, 507],
                [349, 510],
                [352, 369],
                [330, 366],
                [330, 366],
            ]
        ),
    ]

    # add polygons
    shapes_layer = viewer.add_shapes(
        polygons,
        shape_type='polygon',
        edge_width=5,
        edge_color='coral',
        face_color='royalblue',
        name='shapes',
    )

    # Send local variables to the console
    viewer.update_console(locals())

from skimage import data
from skimage import filters
from skimage import segmentation
from skimage import morphology
from skimage import io
import time
import napari
import numpy as np
np.set_printoptions(threshold=np.inf)
my_filetypes = [('all files', '.*'),('Image files', '.tif')]
root = tk.Tk()
root.withdraw()

filez = filedialog.askopenfilename(parent = root, title='Please Select a File', filetypes = my_filetypes)

coins = io.imread(filez)

def sobel_plane_by_plane(img):
    sobel_array = np.zeros(img.shape)
    for n in range(img.shape[0]):
        im_plane = img[n,:,:]
        sobel_array[n,:,:] = filters.sobel(im_plane)
    
    return sobel_array

def watershed_plane_by_plane (sobel_im,marker):
    watershed_im = np.zeros(sobel_im.shape)

    for n in range(sobel_im.shape[0]):
        im_plane = sobel_im[n,:,:]
        watershed_im[n,:,:] = segmentation.watershed(im_plane, markers=markers)
    
    return watershed_im
    
with napari.gui_qt():
   #coins = data.coins()

   viewer = napari.view_image(coins, name='coins', scale = [2.27,1,1])

   edges =sobel_plane_by_plane(coins)

   edges_layer = viewer.add_image(edges, colormap='magenta', blending='additive',scale = [2.27,1,1])

   pts_layer = viewer.add_points(size=5)
   pts_layer.mode = 'add'
   input("Press Enter to continue...")
# annotate the background and all the coins, in that order

   coordinates = pts_layer.data
  
   coordinates_int = np.round(coordinates)[:,1:3].astype(int)
   print(coordinates_int)
   markers_raw = np.zeros_like(coins[8])
   markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(coordinates))
   print(markers_raw)
# raw markers might be in a little watershed "well".
   markers = morphology.dilation(markers_raw, morphology.disk(5))

   segments = segmentation.watershed(edges, markers=markers)

   labels_layer = viewer.add_labels(segments - 1,scale = [2.27,1,1])  # make background 0
'''
'''
import numpy as np 
  
# curve-fit() function imported from scipy 
from scipy.optimize import curve_fit 
  
from matplotlib import pyplot as plt 
  
# numpy.linspace with the given arguments 
# produce an array of 40 numbers between 0 
# and 10, both inclusive 
x = np.linspace(0, 10, num = 40) 
  
  
# y is another array which stores 3.45 times 
# the sine of (values in x) * 1.334.  
# The random.normal() draws random sample  
# from normal (Gaussian) distribution to make 
# them scatter across the base line 
y = 3.45 * np.sin(1.334 * x) + np.random.normal(size = 40) 
  
# Test function with coefficients as parameters 
def test(x, a, b): 
    return a * np.sin(b * x) 
  
# curve_fit() function takes the test-function 
# x-data and y-data as argument and returns  
# the coefficients a and b in param and 
# the estimated covariance of param in param_cov 
param, param_cov = curve_fit(test, x, y) 
  
  
print("Sine funcion coefficients:") 
print(param) 
print("Covariance of coefficients:") 
print(param_cov) 
  
# ans stores the new y-data according to  
# the coefficients given by curve-fit() function 
#ans = (param[0]*(np.sin(param[1]*x))) 
ans = test(x, *param)
Below 4 lines can be un-commented for plotting results  
using matplotlib as shown in the first example. 
fig,ax = plt.subplots()
plt.plot(x, y, 'o', color ='red', label ="data") 
plt.plot(x, ans, '--', color ='blue', label ="optimized data") 
ax.set_xscale('log')
plt.legend() 
plt.show() 
'''


import matplotlib.pyplot as plt
import mplcursors
import numpy as np
x = np.linspace(0, 10, 100)

fig, ax = plt.subplots()

# Plot a series of lines with increasing slopes.
lines = []
for i in range(1, 20):
    ax=plt.plot(x, i * x, label=str(i))

mplcursors.cursor( highlight=True,hover=True)

plt.show()