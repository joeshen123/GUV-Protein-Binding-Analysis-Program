# Analysis Program for analyzing GUV geometry and protein bindings in Time Lapse 3D Confocal Stacks

# Brief Overview

**Python** programs for analyzing GUV geometry (e.g. Volume, radius) and protein bindings in confocal time lapse data (It also works with single time point). Generally, the program accepts input of a 4D confocal stack plus time (TZYX). But it can also accept single z image (TYX). The output are a csv file listing all the measurements and a hdf5 object that contains the same info as csv file and is used by downstream programs to analyse and make plots. Below are **schematics** illustrating the pipeline workflow:  

## Workflow diagrams for analyzing radius and protein binding changes across time
![](Pipeline%20Images/GUV%20Analysis%20Pick%20Middle%20Frame%20Workflow.png)

## Workflow diagrams for analyzing GUV surface area and volume changes accross time
![](Pipeline%20Images/GUV%20Analysis%203D%20Pipeline.png)

The program has both 2D and 3D components. The goal of 2D component is to extract protein recruitment intensity and radius from image data. We use protein binding intensity on the mid slice as an indicator of protein recuruitment since most protein bindings happen on mid slice of GUV. And we use the radius of mid slice as the radius of GUV. In order to do that, we first let user to draw a line to select the GUV they want to analyze. Then the program will crop the GUV based on user's selection and perform intensity normalization followed by gaussian blur. In the next step, the program will perform marker-based watershed algorithm on different z slice to segment the GUV ring and measure their area. The program select the segmentation with largest area as mid-slice. All these steps are performed in GUV Channel. Lastly, we can use the segmented ring of GUV channel to analyze protein binding intensity on the protein channel and plot the changes as a function of time.

The 3D component is very similar to the 2D component. The difference is that instead of performing markers-based watershed algorithm slice by slice, we perform watershed on 3D space and segment GUV as a 3D object rather than accumulation of multiple 2D z stacks. After obtaining the 3D segmented object, we can analyze GUV volume/surface area and plot the changes accross time.
