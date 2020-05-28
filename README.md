# Analysis Program for analyzing GUV geometry and protein bindings in time lapse 3D confocal stacks

# Brief Overview

**Python** programs for analyzing GUV geometry (e.g. Volume, radius) and protein bindings in confocal time lapse data (It also works with single time point). Generally, the program accepts input of a 4D confocal stack plus time (TZYX). But it can also accept single z image (TYX). The output are a csv file listing all the measurements and a hdf5 object that contains the same info as csv file and is used by downstream programs to analyse and make plots. Below are **schematics** illustrating the pipeline workflow:  

## Workflow diagrams for analyzing radius and protein binding changes across time (2D Version)
![](Pipeline%20Images/GUV%20Analysis%20Pick%20Middle%20Frame%20Workflow.png)

## Workflow diagrams for analyzing GUV surface area and volume changes accross time (3D Version)
![](Pipeline%20Images/GUV%20Analysis%203D%20Pipeline.png)

**2D Version Overview:** 
The goal of 2D component is to extract protein recruitment/binding intensity and radius from image data (Image data is comprised of two channels. One is GUV channel, which is used for labelling liposome. Another one is protein channel, which is used for monitoring protein -GUV recruitment. Intensity is obtained through protein channel, and the radius is obtained through GUV channel). We use protein binding intensity on the mid slice as an indicator of protein recuruitment since most protein bindings happen on mid section of GUV. And we use the radius of mid slice as the radius of GUV. In order to do that, we first let users to draw a line on the specific GUV they want to analyze. Then the program will crop the GUV based on user's predefined line and perform intensity normalization followed by gaussian blur. In the next step, the program will perform marker-based watershed algorithm on different z slice to segment the GUV ring and measure their area. The program select the segmented ring with largest area as mid-slice. All these steps are performed in GUV Channel. Lastly, we can use the segmented ring of GUV channel to analyze protein binding intensity on the protein channel as n indicator of recruitment and plot the changes as a function of time.

**3D Version Overview:**
The 3D component is very similar to the 2D component. The difference is that instead of performing markers-based watershed algorithm slice by slice, we perform watershed on 3D space and segment GUV as a 3D object rather than accumulation of multiple 2D z stacks. After obtaining the 3D segmented object, we can analyze GUV volume/surface area and plot the changes accross time.

# Program Components
The bulk of this program is in **Core Programs** directory. Lists below give brief introductions of programs within the directory.

* **Image_Import.py**: Import and Convert image from .ND2 tO .HDF5

* **GUV_Anaysis_Module**: Contain all objects and functions to execute analysis pipelines

* **pre_processing_utils.py**: Contain intensity normalization and gaussian blur functions that are used in GUV_Analysis_Module. It is adapted and modified from **aics-segmentation** (https://github.com/AllenInstitute/aics-segmentation) 

* **utils.py**: Contain other image operation functions modified from **aics-segmentation** that is used in analysis pipeline.

* **Vesicle_Analysis_Execution_2.0.py**: 2D Version of the analysis pipelines

* **Vesicle_Analysis_Execution_2.0_3d.py**: 3D Version of the analysis pipelines

* **Vesicle_Analysis_Execution_2.0_Single.py**: A simplified 2D Version of the analysis pipelines. Input is a single z plane image stack (TYX). Because it is only single z stack, the step of finding mid-slice is skipped. Segmentation is directly applied on GUV channel.

# Installation
I tested the program in both Windows 7 and Mac OS Catalina system (Majority of my work is performed on Mac OS Catalina). The program uses Python 3.7.

## Step by Step Installation Guidance

> 1. Download the [environment.yml](environment.yml) file from the repository either through copy/paste to text editor of choice or git clone the whole repository into local computer.

> 2. Install Conda/Miniconda under the [instruction](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) from the website. 

> 3. After successfully install conda, go to the directory where **environment.yml** locate and run **conda env create -f environment.yml**. It may take several minutes to install all the dependancy for the program

> 4. Now we can activate the specific environment by running **conda activate guv_pipeline_paper**. You should see **(guv_pipline_paper)** on the left of terminal.

>5. Now you can go into **Core Programs** directory and run **python Vesicle_Analysis_Execution_2.0.py** or **python Vesicle_Analysis_Execution_2.0_3d.py** or **python Vesicle_Analysis_Execution_2.0_Single.py** to start the program. Congratulations!

# Requirements
