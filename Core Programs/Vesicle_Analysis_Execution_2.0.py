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
from GUV_Analysis_Module import *
import napari
import time
import matplotlib

#matplotlib.use('Qt4Agg')

#Ignore warnings issued by skimage through conversion to uint8
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

print(image.shape)

intensity_image = f['488 Channel'][:]

#image = image + intensity_image

time_len = simpledialog.askinteger("Input", "How long is the movie in minutes? ",
                                                 parent=root,
                                                 minvalue=0, maxvalue=1000)

# function to extract center point and line of scanning radius for image processing
def pt_dist_extractor(line_data_list):
    point_list = []
    line_list = []
    z_list = []
    for num in range(len(line_data_list)):

        pt = line_data_list[num]
        #print(pt)
        x = pt[0][3]
        y = pt[0][2]
        z = pt[0][1]

        x1 = pt[1][3]
        y1 = pt[1][2]

        point = np.array((x,y))
        point2 = np.array((x1,y1))

        length = np.linalg.norm(point2-point)
        point_list.append(point)
        line_list.append(length)
        z_list.append(z)

    return (point_list, line_list,z_list)

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(image, name='561', colormap='red',blending = 'opaque', scale = [4,1,1])
    viewer.add_image(intensity_image,name='488',colormap='green',blending = 'additive', scale = [4,1,1])
    line_layer = viewer.add_shapes(shape_type='line', edge_width=1.5, face_color = 'blue', edge_color = 'b', scale = [4,1,1])

    # Stop the program for updating shape layer
    input("Press Enter to continue...")
    line_data = line_layer.data

    point_list, line_list,z_list= pt_dist_extractor(line_data)
    Analysis_Stack = Image_Stacks(image,intensity_image,point_list,line_list,z_list,2000,time_len)
    Analysis_Stack.tracking_multiple_circles()
    viewer.add_image(Analysis_Stack.binary_shell_list,name='Mid Plane Contour', blending = 'additive', scale = [4,1,1])
    


GUV_Post_Analysis_df_list = []
GUV_Post_Analysis_df_list += Analysis_Stack.stats_df_list
Pandas_list_plotting(GUV_Post_Analysis_df_list, 'Normalized Intensity',marker='o')
Pandas_list_plotting(GUV_Post_Analysis_df_list,'Radius',marker='o')


del_answer = messagebox.askyesnocancel("Question","Do you want to delete some measurements?")

while del_answer == True:
        delete_answer = simpledialog.askinteger("Input", "What number do you want to delete? ",
                                                 parent=root,
                                                 minvalue=0, maxvalue=100)


        GUV_Post_Analysis_df_list.pop(delete_answer)
        Pandas_list_plotting(GUV_Post_Analysis_df_list, 'Normalized Intensity',marker='o')
        Pandas_list_plotting(GUV_Post_Analysis_df_list,'Radius',marker='o')
        del_answer = messagebox.askyesnocancel("Question","Do you want to delete more measurements?")



File_save_names = '.'.join(file_name.split(".")[:-1])
list_save_name='{File_Name}_analysis'.format(File_Name = File_save_names)

# save as csv
csv_save_name = list_save_name + '.csv'
pd.concat(GUV_Post_Analysis_df_list).to_csv(csv_save_name)


#save as hdf5 file
n = 1
save_name = list_save_name +'.hdf5'
for df in GUV_Post_Analysis_df_list:
    key_tag = 'df_' + str(n)
    df.to_hdf(save_name, key = key_tag, complib='zlib', complevel=5)
    n += 1

#Save the segmentation Binary Result for downstream analysis
seg_save_name='{File_Name}_segmentation_result_2D.hdf5'.format(File_Name = File_save_names)

with h5py.File(seg_save_name, "w") as f:
      f.create_dataset('Segmentation_Binary_Result', data = Analysis_Stack.binary_shell_list, compression = 'gzip')
   
