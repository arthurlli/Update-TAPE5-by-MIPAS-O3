# Update-TAPE5-by-MIPAS-O3
A program to update ozone concnetrations in the user defined atmosphere of TAPE5 (configuration file of LBLRTM) with MIPAS O3 data over nearest observations of Tokyo.

# Use case
If you are working on satellite retrieval and would like to use [Line-By-Line Radiative Transfer Model (LBLRTM)](https://github.com/AER-RC/LBLRTM) as forward model, this code may help you to update O3 concentrations in user defined atmosphere using MIPAS O3 data product ([López-Puertas et al., 2023](https://doi.org/10.5194/amt-16-5609-2023)). 

# Before running main.py
Please download the MIPAS O3 data product beforehand and store it to "./data/" folder.<br>
Dataset: MIPAS ozone retrieval version 8: middle atmosphere measurements; the data set (https://dx.doi.org/10.35097/1803)<br>

# Summary
Summary: the main program does the following<br>
    1) read all netCDF4 files of MIPAS O3<br>
    2) compute mean O3 values after interpolation to unified pressure levels<br>
    3) read base TAPE5 model atmosphere and update it with mean MIPAS O3 values<br>

Input: <br>
    1) netCDF4 files (.nc) under "./data/" directory<br>
    2) base TAPE5 atmosphere: TAPE5-atm-base.txt<br>

Output:<br>
    1) JPGs plots of MIPAS O3 data over Tokyo<br>
    2) modified TAPE5 atmosphere: TAPE5-atm-MIPAS.txt<br>

# Side note
* You may want to run check-xx.py in "./check-attributes" folder to check attributes of MIPAS O3 dataset.<br>
* The "target" attribute stores O3 concentrations.
* You can change the latitude and longitude for other cities (default is Tokyo).
