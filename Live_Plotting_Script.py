from matplotlib import gridspec
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from pandas import HDFStore
import pandas as pd
from skimage import io
import numpy as np
from matplotlib.pyplot import cm
from matplotlib.animation import ArtistAnimation
from matplotlib.colors import LinearSegmentedColormap

root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('csv files', '.csv')]

c2_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)
'''
store = HDFStore(c2_path)

for key in store.keys():
    print(key)
    df = store[key]
'''
df = pd.read_csv(c2_path)

df = df.drop(df.columns[0], axis = 1)

print(df)


my_filetypes = [('all files', '.*'),('image files', '.tif')]
guv_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

guv = io.imread(guv_path)

int_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

inten = io.imread(int_path)

#Make a colormap like Imagej Red Channel
cdict2 = {'red':  ((0.0, 0.0, 0.0),   # <- at 0.0, the red component is 0
                   (0.5, 0.5, 0.5),   # <- at 0.5, the red component is 1
                   (1.0, 1.0, 1.0)),  # <- at 1.0, the red component is 0

         'green': ((0.0, 0.0, 0.0),   # <- etc.
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
         }

Red = LinearSegmentedColormap('Red', cdict2)

#Make a colormap like Imagej Green Channel
cdict1 = {'red':  ((0.0, 0.0, 0.0),   # <- at 0.0, the red component is 0
                   (0.5, 0.0, 0.0),   # <- at 0.5, the red component is 1
                   (1.0, 0.0, 0.0)),  # <- at 1.0, the red component is 0

         'green': ((0.0, 0.0, 0.0),   # <- etc.
                   (0.5, 0.5, 0.5),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
         }

Green = LinearSegmentedColormap('Green', cdict1)


# Define a function to live plot the protein intesnity data
def live_plotting_intensity(df,im1,im2):
       #shortest_len = min(df1.shape[0],df2.shape[0],df3.shape[0])
       c2_Intensity_data = df['Normalized GFP intensity'].tolist()
       #lox5_Intensity_data = lox5_df['GFP intensity'].tolist()[0:shortest_len]
       #pkc_Intensity_data = pkc_df['GFP intensity'].tolist()[0:shortest_len]
       c2_radius = df['radius_micron'].tolist()

       Time_point = df['Time Point'].tolist()

       c2_Intensity_with_Time = []
       c2_Intensity_with_Time.append(Time_point)
       c2_Intensity_with_Time.append(c2_Intensity_data)
       c2_Intensity_with_Time = np.array(c2_Intensity_with_Time)

       c2_radius_time = []
       c2_radius_time.append(Time_point)
       c2_radius_time.append(c2_radius)
       c2_radius_time = np.array(c2_radius_time)

       gs = gridspec.GridSpec(2,2)
       fig = plt.figure(figsize = (10,10))
       ax1 = fig.add_subplot(gs[0:1,0:1])
       ax2 = fig.add_subplot(gs[0:1,1:2])
       ax3 = fig.add_subplot(gs[1:2,:])

       ax1.set_title('GUV Channel',fontsize=22)
       ax2.set_title('Protein Binding Channel',fontsize=22)

       ax3.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold')
       ax3.set_ylabel('Protein Membrane Bindings (AU)',fontsize = 16, fontweight = 'bold', color = 'r')
       
       ax3.tick_params(axis = 'y', labelcolor = 'k',labelsize=16)
       ax3.tick_params(axis = 'x', labelcolor = 'k',labelsize=16)
       
       ax4 = ax3.twinx()

       ax4.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold')
       ax4.set_ylabel('Radius Changes (um)', fontsize = 16,fontweight = 'bold',color='b')
       
       ax4.tick_params(axis = 'y', labelcolor = 'k',labelsize=16)
       ax4.tick_params(axis = 'x', labelcolor = 'k',labelsize=16)

       ims = []

       for time in range(len(c2_Intensity_data)):
           guv_im = ax1.imshow(im1[time],Red)
           ax1.axis('off')

           binding_im = ax2.imshow(im2[time],Green)
           ax2.axis('off')



           l_one, = ax3.plot(c2_Intensity_with_Time[0,:time],c2_Intensity_with_Time[1,:time],'r-')
           l_two, = ax4.plot(c2_radius_time[0,:time],c2_radius_time[1,:time],'b-')
           ax3.set_ylim(50,380)
           ax4.set_ylim(8.4,9.6)
           ax4.legend((l_one,l_two),('Bindings','Radius'),loc=0, prop={'size': 14})


           ims.append([guv_im,binding_im,l_one,l_two])

       line_ani = ArtistAnimation(fig,ims, interval=50,blit=False,repeat=True)
       plt.tight_layout()

       return line_ani


live_plot = live_plotting_intensity(df, guv, inten)
#plt.show()
movie_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a movie name for saving:",filetypes=[('Movie files', '.mp4')])

live_plot.save(movie_save_name,fps=10)
