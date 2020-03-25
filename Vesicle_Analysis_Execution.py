import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from GUV_Analysis_Module import *
import warnings
from tkinter import simpledialog
from tkinter import Tk, Label, Button, Radiobutton, IntVar
import h5py
from skimage.external import tifffile

#Ignore warnings issued by skimage through conversion to uiquit()nt8
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)
warnings.simplefilter("ignore",FutureWarning)
# Use tkinter to interactively select files to import
root = tk.Tk()
root.withdraw()

GUV_Post_Analysis_df_list = []

my_filetypes = [('all files', '.*'),('Image files', '.hdf5')]

filez = filedialog.askopenfilenames(parent = root, title='Please Select a File', filetypes = my_filetypes)

file_name= root.tk.splitlist(filez)[0]

f = h5py.File(file_name, 'r')

image = f['561 Channel'][:]

intensity_image = f['488 Channel'][:]

image = intensity_image + image

f.close()
# Running Image Analysis Pipelines
Analysis_Stack = Image_Stacks(image,intensity_image, np.array((77.52769934,45.11785543)), np.array((13.733838631289755)))
   
#Analysis_Stack.set_parameter()    
#Analysis_Stack.enhance_blur_first()

#choice = messagebox.askyesno("Question","Do you want to choose manual segmentation?")

#if choice == True:
  #Analysis_Stack.set_points_manual()

#elif choice == False:
  #Analysis_Stack.set_points_automatic()
  
Analysis_Stack.tracking_multiple_circles()
#Analysis_Stack.displaying_circle_movies()

#display_image_sequence(Analysis_Stack.Rendering_Image_Stack,'Display Original Images')

# add circle patch to current measured GUV

GUV_Post_Analysis_df_list += Analysis_Stack.stats_df_list
Pandas_list_plotting(GUV_Post_Analysis_df_list, 'Normalized Intensity')
Pandas_list_plotting(GUV_Post_Analysis_df_list,'Radius')
  
del_answer = messagebox.askyesnocancel("Question","Do you want to delete some measurements?")

while del_answer == True:
  delete_answer = simpledialog.askinteger("Input", "What number do you want to delete? ",
                                 parent=root,
                                 minvalue=0, maxvalue=100)


  GUV_Post_Analysis_df_list.pop(delete_answer)
  Pandas_list_plotting(GUV_Post_Analysis_df_list, 'Intensity')
  Pandas_list_plotting(GUV_Post_Analysis_df_list,'Radius')
  del_answer = messagebox.askyesnocancel("Question","Do you want to delete more measurements?")
      

save_answer = messagebox.askyesnocancel("Question","Do you want to save the measurement?")

if save_answer == True:

   File_save_names = file_name.split(".")[0]
   list_save_name='{File_Name}_analysis'.format(File_Name = File_save_names)
#middle_stack = Analysis_Stack.middle_slice_stack
#middle_stack_intensity = Analysis_Stack.middle_slice_stack_intensity

#tifffile.imsave('C:/Users/joeshen/Desktop/Middle_frame_stack',middle_stack.astype('uint16'),imagej=True,bigtiff=True,metadata={'axes': 'TYX'})
#tifffile.imsave('C:/Users/joeshen/Desktop/Middle_frame_stack_intensity',middle_stack_intensity.astype('uint16'),imagej=True,bigtiff=True,metadata={'axes': 'TYX'})

   
   csv_save_name = list_save_name + '.csv'
   pd.concat(GUV_Post_Analysis_df_list).to_csv(csv_save_name)
  
   n = 1
   for df in GUV_Post_Analysis_df_list:
     save_name = list_save_name +'.hdf5'
     key_tag = 'df_' + str(n)
     df.to_hdf(save_name, key = key_tag, complib='zlib', complevel=5)
     n += 1
  

