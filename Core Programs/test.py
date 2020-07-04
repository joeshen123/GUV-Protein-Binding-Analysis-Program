"""
Display a 2D surface
"""

import numpy as np
from skimage import data
import napari

"""
Display a 3D surface
"""

import numpy as np
from skimage import data
import napari


with napari.gui_qt():
    # create the viewer and window
    viewer = napari.Viewer(ndisplay=3)

    data = np.array([[0,0, 0, 0], [0,0, 20, 10], [0,10, 0, -10], [0,10, 10, -10],[1,0, 0, 2], [1,0, 20, 15], [1,10, 0, -20], [1,10, 10, -18]])
    faces = np.array([[0,0, 1, 2], [0,1, 2, 3],[1,0, 1, 2], [1,1, 2, 3]])
    values = np.linspace(0, 1, 4)
    values = np.concatenate([values,values]).reshape(1,8)
    print(values.shape)
    # add the surface
    layer = viewer.add_surface((data, faces, values))