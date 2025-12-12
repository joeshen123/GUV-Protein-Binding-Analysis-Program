# Analysis Program for analyzing GUV geometry and protein bindings in time lapse 3D confocal stacks

**This repository includes two standalone components:**

1. Image-analysis program — extracts GUV membrane geometry (radius, surface area, volume) and quantifies protein adsorption to the membrane.

2. Fitting program — performs Langmuir adsorption–isotherm fitting on equilibrium binding data to determine membrane affinity ($K_d'$).

This code has been used in the following publications:

**A synergy between mechanosensitive calcium- and membrane-binding mediates tension-sensing by C2-like domains.**
**<br>DOI: https://www.pnas.org/doi/10.1073/pnas.2112390119**


**Endoplasmic reticulum disruption stimulates nuclear membrane mechanotransduction.**
**<br>DOI: https://www.nature.com/articles/s41556-025-01820-9**


# Brief Overview of Image-analysis program

**Python** programs for analyzing GUV geometry (e.g. Volume, radius) and protein bindings in confocal time lapse data (It also works with single time point). Generally, the program accepts input of a 4D confocal stack plus time (TZYX). But it can also accept single z image (TYX). The output are a csv file listing all the measurements and a hdf5 object that contains the same info as csv file and is used by downstream programs to analyse and make plots. Below are **schematics** illustrating the pipeline workflow:  

## Workflow diagrams for analyzing radius and protein binding changes across time (2D Version)
![](Pipeline%20Images/GUV%20Analysis%20Pick%20Middle%20Frame%20Workflow.png)

## Workflow diagrams for analyzing GUV surface area and volume changes accross time (3D Version)
![](Pipeline%20Images/GUV%20Analysis%203D%20Pipeline.png)

**2D Version Overview:** 
The goal of 2D component is to extract protein recruitment/binding intensity and radius from image data (Image data is comprised of two channels. One is GUV channel, which is used for labelling liposome. Another one is protein channel, which is used for monitoring protein -GUV recruitment. Intensity is obtained through protein channel, and the radius is obtained through GUV channel). We use protein binding intensity on the mid slice as an indicator of protein recuruitment since most protein bindings happen on mid section of GUV. And we use the radius of mid slice as the radius of GUV. In order to do that, we first let users to draw a line on the specific GUV they want to analyze. Then the program will crop the GUV based on user's predefined line and perform intensity normalization followed by gaussian blur. In the next step, the program will perform marker-based watershed algorithm on different z slice to segment the GUV ring and measure their area. The program select the segmented ring with largest area as mid-slice. All these steps are performed in GUV Channel. Lastly, we can use the segmented ring of GUV channel to analyze protein binding intensity on the protein channel as n indicator of recruitment and plot the changes as a function of time.

**3D Version Overview:**
The 3D component is very similar to the 2D component. The difference is that instead of performing markers-based watershed algorithm slice by slice, we perform watershed on 3D space and segment GUV as a 3D object rather than accumulation of multiple 2D z stacks. After obtaining the 3D segmented object, we can analyze GUV volume/surface area and plot the changes accross time.

## Program Components
The bulk of this program is in **Core Programs** directory. Lists below give brief introductions of programs within the directory.

* **Image_Import.py**: Import and Convert image from .ND2 to .HDF5 where all images are stored as numpy arrays.

* **GUV_Anaysis_Module**: Contain all objects and functions to execute analysis pipelines

* **pre_processing_utils.py**: Contain intensity normalization and gaussian blur functions that are used in GUV_Analysis_Module. It is adapted and modified from **aics-segmentation** (https://github.com/AllenInstitute/aics-segmentation) 

* **utils.py**: Contain other image operation functions modified from **aics-segmentation** that is used in analysis pipeline.

* **Vesicle_Analysis_Execution_2.0.py**: 2D Version of the analysis pipelines

* **Vesicle_Analysis_Execution_2.0_3d.py**: 3D Version of the analysis pipelines

* **Vesicle_Analysis_Execution_2.0_Single.py**: A simplified 2D Version of the analysis pipelines. Input is a single z plane image stack (TYX). Because it is only single z stack, the step of finding mid-slice is skipped. Segmentation is directly applied on GUV channel.

## Installation
I tested the program in both Windows 7 and Mac OS Catalina system (Majority of my work is performed on Mac OS Catalina). The program uses Python 3.7.

## Step by Step Installation Guidance

1. Download the [environment.yml](environment.yml) file from the repository either through copy/paste to text editor of choice or git clone the whole repository into local computer.

2. Install Conda/Miniconda under the [instruction](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) from the website. 

3. After successfully install conda, go to the directory where **environment.yml** locate and run: 

```bash
conda env create -f environment.yml 
```
It may take several minutes to install all the dependancy for the program.

4. Now we can activate the specific environment by running:

```bash
conda activate guv_pipeline 
```
You should see **(guv_pipline)** on the left of terminal.

5. Now you can go into **Core Programs** directory and run one of the following commands depends on which program you want to use:
```bash
python Vesicle_Analysis_Execution_2.0.py
   
python Vesicle_Analysis_Execution_2.0_3d.py
   
python Vesicle_Analysis_Execution_2.0_Single.py
```
**Congratulations!**

# Brief Overview of Fitting Program

To quantitatively measure equilibrium adsorption of peripheral membrane proteins onto GUV membranes, we fit the equilibrium membrane binding fluorescence vs. protein concentration data using a Langmuir adsorption isotherm with a Hill expansion.


## ⚙️ Model Overview

To describe protein adsorption to the GUV membrane, we use the classical Langmuir model with Hill expansion:

$$
B_{\text{bound}} = \frac{B_{\max} \ c^{H}}{c^{H} + {K_d'}^{H}}
$$

Where:

- **$B_{\text{bound}}$** — measured equilibrium fluorescence on the GUV membrane  
- **$c$** — protein concentration in solution  
- **$B_{\max}$** — maximum binding intensity (after saturation)  
- **$K_d'$** — apparent dissociation constant (proxy for affinity)  
- **$H$** — Hill coefficient (cooperativity)

This formulation captures:

- **Binding affinity** via ($K_d'$) 
- **Maximum adsorption** via ( $B_{\max}$ ) 
- **Cooperative behavior** via the Hill coefficient ($H$)


## Program Components

The fitting program is stored in the **Langmuir Fitting** directory

- **ALPS 1-2-eGFP Langmuir Fitting.m**: Scripts used to run the fitting and plotting of the graph.

- **FitLangmuir.m**: Store the function to fit the data and generate the plots

- **createFitLM_hill.m**: Store the Langmuir Model

The fitting program is written in MATLAB and it is tested in MATLAB R2025a with Curve Fitting Toolbox installed (Version 25.1). No installation is required, everything is provided in the program.


# Demo and Walkthrough

- ### We have provided the demo data and screenshots of the key steps to demonstrate the analysis workflow for both the Image Analysis Program and the Fitting Program.
  - The demo data is stored in the **Demo** directory.

    - Image Analysis Program: The demo dataset is a cropped subset of the original timelapse movies and includes only five timepoints.

    - Fitting Program: The demo dataset consists of a single .mat file containing ALPs1–2–eGFP GUV membrane equilibrium-binding measurements.

- ### Below are the walkthrough of the Image Analysis program.
   
   - **Step 1: Run Vesicle_Analysis_Execution_2.0.py**
  ![Step1](Demo%20Step/Image%20Analysis/Step1.png)
    
   - **Step 2: Input the length of the movie (For demo purpose, enter 5)**
  ![Step 2](Demo%20Step/Image%20Analysis/Step%202.png)

   - **Step 3: Draw a line from the center of GUV to the edge**
  ![Step 3](Demo%20Step/Image%20Analysis/Step%203.png)

   - **Step 4: Program is running**
  ![Step 4](Demo%20Step/Image%20Analysis/Step%204.png)

   - **Step 5: Use Napari viewer to visualize the segmented contours drawn on the mid slice of GUV**
  ![Step 5](Demo%20Step/Image%20Analysis/Step5.png)
  
   - **Step 6: View the final quantitative plots of radius and protein adsorption obtained from the analysis**
  ![Step 6](Demo%20Step/Image%20Analysis/Step6.png)
  ![Step 7](Demo%20Step/Image%20Analysis/Step7.png)

   - **Step 7: Choose to delete specific GUVs (e.g. broken segmentation) before saving the segmentation masks and analyzed measurements**
   ![Step 8](Demo%20Step/Image%20Analysis/Step8.png)


- ### Below are the walkthrough of the Fitting program.
   
   - **Step 1: Load the ALPS1-2-eGFP equilibrium binding measurements into MATLAB Woprkspace**
  ![Step1](Demo%20Step/Fitting/Step1.png)
    
   - **Step 2: Run the ALPS 1-2-eGFP Langmuir Fitting.m for fitting and plotting**
  ![Step 2](Demo%20Step/Fitting/Step2.png)

   - **Step 3: View the plots of equilibrium binding measurements with Langmuir isotherm-fitted curves. Details of $K_d'$, $B_{\max}$ and $H$ with associated 95% C.I. can be found in the variables created in the MATLAB workspace**
  ![Step 3](Demo%20Step/Fitting/Step3.png)


# Requirements and Acknowledgement
I am really grateful for the open source community of python scientific computing. Without these great tools, it is impossible for me to develop the analysis pipeline. Here are the major softwares/tools used in the pipeline:

* [Scikit-Image](https://scikit-image.org/)

* [nd2reader](https://rbnvrw.github.io/nd2reader/)

* [aics-segmentation](https://github.com/AllenInstitute/aics-segmentation)

* [Napari](https://github.com/napari/napari)

## ❓ Troubleshooting

If you have any questions, feel free to email joeshenz123@gmail.com



