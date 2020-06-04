# Program to roughly count the number of GUV in a image. Use Cellpose as a segmenter. Details are here: http://www.cellpose.org/static/docs/index.html

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import warnings
from tkinter import simpledialog
from tkinter import Tk, Label, Button, Radiobutton, IntVar
import h5py
import numpy as np
from skimage import transform
from scipy.ndimage import zoom
from skimage.measure import label, regionprops
import time
import matplotlib
import os
import glob
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from skimage import io

#matplotlib.use('Qt4Agg')

# Save specific font that can be recognized by Adobe Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#Ignore warnings issued by skimage through conversion to uint8
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)
warnings.simplefilter("ignore",FutureWarning)
# Use tkinter to interactively select files to import
root1 = tk.Tk()
root1.withdraw()

GUV_Post_Analysis_df_list = []

my_filetypes = [('all files', '.*'),('Image files', '.hdf5')]

root1.directory = filedialog.askdirectory(parent = root1)

os.chdir(root1.directory)

path = 'MIP_Original'
try:
    os.mkdir(path)
except FileExistsError:
    pass

img_names = glob.glob('*.hdf5' )


for img in img_names:
   label = path + '/' + img + '.tiff'
   f = h5py.File(img, 'r')
   GUV_image_stk = f['561 Channel'][0]
   GUV_image_max= np.max(GUV_image_stk, axis=0)
   
   io.imsave(label, GUV_image_max)
