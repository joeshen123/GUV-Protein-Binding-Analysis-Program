# Analysis Program for analyzing GUV geometry and protein bindings in Time Lapse 3D Confocal Stacks

# Brief Overview

**Python** programs for analyzing GUV geometry (e.g. Volume, radius) and protein bindings in confocal time lapse data (It also works with single time point). Generally, the program accepts input of a 4D confocal stack plus time (TZYX). But it can also accept single z image (TYX). The output are a csv file listing all the measurements and a hdf5 object that contains the same info as csv file and is used by downstream programs to analyse and make plots. Below are **schematics** illustrating the pipeline workflow:  

## Pipeline for analyzing radius and protein binding changes across time
![](Pipeline%20Images/GUV%20Analysis%20Pick%20Middle%20Frame%20Workflow.png)


