from nd2reader.reader import ND2Reader
import numpy as np
import time
from tkinter import ttk
from tkinter import simpledialog
from tkinter import filedialog
import tkinter as tk
from skimage.external import tifffile
import matplotlib.pyplot as plt
from skimage import io
import warnings
import h5py

#Ignore warnings issued by skimage through conversion to uint8
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)

# Use tkinter to interactively select files to import
root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('Movie files', '.nd2')]

Image_Stack_Path = filedialog.askopenfilename(title='Please Select a Movie', filetypes = my_filetypes)

print (Image_Stack_Path)
# Define a function to convert time series of ND2 images to a numpy list of Max Intensity Projection

Image_Sequence = ND2Reader(Image_Stack_Path)
print(Image_Sequence.metadata["channels"])