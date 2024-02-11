"""
This script plot MIPAS Ozone data over Tokyo and computes mean Ozone values
according to the input "TAPE5.base" 

Author: Arthur Li
Version: 1.0
Last update: 2024/2/12 

"""


# note: run check-xx.py to see attributes!
# "target" contains all ozone mole fration

import netCDF4 as nc
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

import os
import glob
from netCDF4 import Dataset
from scipy.interpolate import interp1d

def main():
    """
    Summary: the main program does the following
        1) read all netCDF4 files of MIPAS O3
        2) compute mean O3 values after interpolation to unified pressure levels
        3) read base TAPE5 model atmosphere and update it with mean MIPAS O3 values
        
    Input: 
        1) netCDF4 files (.nc) under "./data/" directory
        2) base TAPE5 atmosphere: TAPE5-atm-base.txt
        
    Output:
        1) modified TAPE5 atmosphere: TAPE5-atm-MIPAS.txt
        
    Note: copy and paste mannually the modified TAPE5 atmosphere to base TAPE5 for further processing
    Note 2: MIPAS O3 in PPM, so unit configuration of O3 in TAPE5 is also updated to PPM
    """
    def read_netCDF4(file_path: str):
        """
        Summary: reading MIPAS O3 data

        Args:
            file_path (str): path of source file

        Returns:
            dt (nc._netCDF4.Dataset): dataset for further processing
        """
        dt = nc.Dataset(file_path, 'r')
        return dt

    def get_attribute(att_name: str, dt: nc._netCDF4.Dataset):
        """
        Summary: read attributes of inputed nc file

        Args:
            att_name (str): attribute name
            dt (nc._netCDF4.Dataset): netCDF4 dataset 

        Returns:
            target_variable: sliced data by att_name 
        """
        target_variable = dt.variables[att_name]
        return target_variable
    
    def find_nearest_lat_lon(
            latitude: float, longitude: float,
            latitudes: np.array, longitudes: np.array):
        """
        Summary: find nearest latitude and longitude in each arrays

        Args:
            latitude (float): targeted latitude
            longitude (float): targeted longitude
            latitudes (np.array): array that stores all latitudes
            longitudes (np.array): array that stores all longitudes

        Returns:
            min_idx (int): index of the minimum distance
        """
        # Use Haversine formula to calculate distances
        distances = cdist([(latitude, longitude)], list(zip(latitudes, longitudes)))
        # Find the indices of the minimum distance
        min_idx = np.argmin(distances)
        return min_idx
    
    def getTKY_idx(dataset: nc._netCDF4.Dataset):
        """
        Summary: get index of minimum distance for Tokyo

        Args:
            dataset (nc._netCDF4.Dataset): netCDF4 dataset 

        Returns:
            tokyo_lat_idx (int): index of minimum distance of tokyo by latitude
            tokyo_lon_idx (int): index of minimum distance of tokyo by longitude
            latitude_values[tokyo_lat_idx] (float): latitude value of minimum distance of tokyo
            longitude_values[tokyo_lon_idx] (float): longitude value of minimum distance of tokyo 
        """
        # Tokyo coordinates (replace with the actual coordinates)
        tokyo_latitude = 35.6895
        tokyo_longitude = 139.6917
        # Extract latitude and longitude variables
        latitude_values = dataset.variables['latitude'][:]
        longitude_values = dataset.variables['longitude'][:]
        # Find the indices of the nearest latitude and longitude to Tokyo
        tokyo_lat_idx = find_nearest_lat_lon(tokyo_latitude, tokyo_longitude, latitude_values, longitude_values)
        tokyo_lon_idx = find_nearest_lat_lon(tokyo_latitude, tokyo_longitude, latitude_values, longitude_values)
        return tokyo_lat_idx,tokyo_lon_idx,latitude_values[tokyo_lat_idx],longitude_values[tokyo_lon_idx]
    
    def check_idx_lat_lon(lat: float, lon: float):
        """
        Summary: check index of latitude and longitude

        Args:
            lat (float): latitude value
            lon (float): longitude value

        Returns:
            boolean: True if they are the same, else False
        """
        if lat==lon:
            print("Lat Lon idx: same")
            return True
        else:
            print("Lat Lon idx: different")
            return False
        
    def check_stats(values: np.array):
        """
        Summary: check statistics of input array

        Args:
            values (np.array): array of float/int values
        Output:
            print out information
        """
        # Calculate statistics
        target_mean = np.mean(values)
        target_std = np.std(values)
        target_min = np.min(values)
        target_max = np.max(values)

        # Display statistics
        print("###########################################")
        print("Statistics for 'target' variable:")
        print(f"  Length of dt: {len(values)}")
        print(f"  Mean: {target_mean}")
        print(f"  Standard Deviation: {target_std}")
        print(f"  Minimum: {target_min}")
        print(f"  Maximum: {target_max}")
        print("###########################################")
        return
    
    def plot_O3(
            dt: np.array, latlon: tuple,
            p: np.array = None, h: np.array = None,
            svn: str = None):
        """
        Summary: plot sample O3 profile with only 1D data

        Args:
            dt (np.array): 1D data of O3
            latlon (tuple): values of latitude and longitude
            p (np.array, optional): 1D data of pressure levels. Defaults to None.
            h (np.array, optional): 1D data of altitudes. Defaults to None.
            svn (str, optional): user defined save name. Defaults to None.
        Output:
            JPG saved as "one-ozone-{p/h}.jpg" to "./"
        """
        # Create a vertical plot
        fig = plt.figure(figsize=(4, 6))
        if p is not None:
            #plt.plot(dt, p, marker='o', linestyle='-', color='b')
            plt.plot(dt, p, marker='x', linestyle='-', color='b')
            plt.yscale('log')
            plt.gca().invert_yaxis()
            saven = 'p'
        elif h is not None:
            plt.plot(dt, h, marker='x', linestyle='-', color='b')        
            saven = 'h'
        # Set plot labels and title
        plt.xlabel('Ozone [VMR 1e-6]')
        plt.ylabel('Pressure [hPa]')
        plt.title(f'Lat. Lon.: {latlon}')
        plt.grid(True)
        plt.tight_layout()
        if svn is None:
            fig.savefig(f'one-ozone-{saven}.jpg',dpi=300)
        else:
            fig.savefig(f'one-ozone-{svn}.jpg',dpi=300)
        return
    
    def plot_all_O3(
            dt: np.array, latlon: tuple, 
            fns: np.array = None, p: np.array = None,
            h: np.array = None):
        """
        Summary: plot O3 profiles with data matrix

        Args:
            dt (np.array): MxN data matrix of ozone (M is year, N is pressure levels)
            latlon (tuple): values of latitude and longitude
            fns (np.array, optional): labels to identify profile. Defaults to None.
            p (np.array, optional): MxN data matrix of pressure levels (same format as dt). Defaults to None.
            h (np.array, optional): MxN data matrix of altitudes (same format as dt). Defaults to None.
        Output:
            JPG named "all-ozone-{h/p}.jpg" to "./"
        """
        # Create a vertical plot
        fig = plt.figure(figsize=(4, 6))
        print(f"Shape dt: {dt.shape}")
        if p is not None:
            print(f"Shape p: {p.shape}")
            for i,x in enumerate(dt):
                #plt.plot(dt, p, marker='o', linestyle='-', color='b')
                plt.plot(x, p[i,:], marker='x', linestyle='-',label=f'{fns[i][19:25]}')
                saven = 'p'
            plt.yscale('log')
            plt.gca().invert_yaxis()
            # set y limit to 0.1 hpa: lower limit by default
            plt.ylim((np.max(p),0.1))
        elif h is not None:
            print(f"Shape h: {h.shape}")
            for i,x in enumerate(dt):
                plt.plot(x, h[i,:], marker='x', linestyle='-',label=f'{fns[i][19:25]}')
                saven = 'h'
        # Set plot labels and title
        plt.legend()
        plt.xlabel('Ozone [ppm]')
        plt.ylabel('Pressure [hPa]')
        plt.title(f'Lat. Lon.: {latlon}')
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(f'all-ozone-{saven}.jpg',dpi=300)
        return
    
    def plot_vertical_mean_O3(
            dt: np.array, latlon: tuple,
            fns: np.array = None, p: np.array = None,
            h: np.array = None, meandt: np.array = None, 
            meandtp: np.array = None,svn: str = None,
            std_newo3: np.array = None):
        """
        Summary: plot vertical mean O3 profile with all O3 profiles as shaded

        Args:
            dt (np.array): MxN data matrix of ozone (M is year, N is pressure levels)
            latlon (tuple): values of latitude and longitude
            fns (np.array, optional): labels to identify profile. Defaults to None.
            p (np.array, optional): MxN data matrix of pressure levels (same format as dt). Defaults to None.
            h (np.array, optional): MxN data matrix of altitudes (same format as dt). Defaults to None.
            meandt (np.array, optional): mean O3 values. Defaults to None.
            meandtp (np.array, optional): mean pressure levels. Defaults to None.
            svn (str, optional): user defined save name. Defaults to None.
            std_newo3 (np.array, optional): standard deviation of the mean O3 values. Defaults to None.
        Output:
            JPG named as "all-ozone-{h/p}.jpg" to "./"
        """
        # Create a vertical plot
        fig = plt.figure(figsize=(3, 5))
        print(f"Shape dt: {dt.shape}")
        if p is not None:
            print(f"Shape p: {p.shape}")
            # else if plot mean O3 data -> add SD bar
            #plt.plot(meandt,meandtp,ls='-',label='Mean',zorder=999,c='red')
            plt.errorbar(meandt,meandtp,xerr=std_newo3,ecolor='red',c='red',zorder=999,label='Mean',barsabove=True)
            for i,x in enumerate(dt):
                #plt.plot(dt, p, marker='o', linestyle='-', color='b')
                plt.plot(x, p, linestyle='-',c='black',alpha=0.2,lw=5)
                saven = 'p' if svn is None else svn
            plt.yscale('log')
            plt.gca().invert_yaxis()
            # set y limit to 0.1 hpa: lower limit by default
            plt.ylim((np.max(p),0.1))
        elif h is not None:
            print(f"Shape h: {h.shape}")
            plt.plot(meandt,meandtp,marker='x',ls='-',label='Mean',zorder=999,c='red')
            for i,x in enumerate(dt):
                plt.plot(x, h, linestyle='-',c='black',alpha=0.2,lw=5)
                saven = 'h'
        # Set plot labels and title
        plt.legend()
        plt.xlabel('Ozone [ppm]')
        plt.ylabel('Pressure [hPa]')
        plt.title(f'Lat. Lon.: {latlon}')
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(f'all-ozone-{saven}.jpg',dpi=300)
        return
    
    def get_date_range_from_filename(path: str = './data/'):
        """
        Summary: obtain range of dates from all file names

        Args:
            path (str, optional): Defaults to './data/'.

        Returns:
            date_range (np.array): array of string of dates 
        """
        all_files = os.listdir(path)
        files_nc = [file for file in all_files if file.endswith('.nc')]
        # remove undisired alphebats
        date_range = np.array([file.replace("MIPAS-E_IMK.","").replace(".V8R_O3_561.nc","") for file in files_nc])
        #print(date_range)
        return date_range
    
    def plot_heatmap_mean_O3(
            dt: np.array, latlon: tuple,
            fns: np.array = None, p: np.array = None,
            h: np.array = None, meandt: np.array =None,
            meandtp: np.array = None, svn: str = None,
            std_newo3: np.array = None):
        """
        Summary: plot heatmap of mean O3 values

        Args:
            dt (np.array): MxN data matrix of ozone (M is year, N is pressure levels)
            latlon (tuple): values of latitude and longitude
            fns (np.array, optional): labels to identify profile. Defaults to None.
            p (np.array, optional): MxN data matrix of pressure levels (same format as dt). Defaults to None.
            h (np.array, optional): MxN data matrix of altitudes (same format as dt). Defaults to None.
            meandt (np.array, optional): mean O3 values. Defaults to None.
            meandtp (np.array, optional): mean pressure levels. Defaults to None.
            svn (str, optional): user defined save name. Defaults to None.
            std_newo3 (np.array, optional): standard deviation of the mean O3 values. Defaults to None.
        Output:
            JPG named as "all-ozone-{h/p}.jpg" to "./"
        """
        # Create a vertical plot
        fig,ax = plt.subplots(figsize=(8, 5))
        print(f"Shape dt: {dt.shape}")
        if p is not None:
            print(f"Shape p: {p.shape}")
            # plot heat map: x is subtract from filenames -> yyyymm
            x = get_date_range_from_filename()
            y = np.copy(p)
            c = ax.pcolormesh(x,y,dt.T)
            plt.yscale('log')
            plt.gca().invert_yaxis()
            # set y limit to 0.1 hpa: lower limit by default
            plt.ylim((np.max(p),0.1))
            saven = 'p' if svn is None else svn
        elif h is not None:
            print(f"Shape h: {h.shape}")

        # Set plot labels and title
        fig.colorbar(c, ax=ax)
        plt.xlabel('Date')
        # Rotate x-axis tick labels
        plt.xticks(rotation=45)
        # only integer ticks are shown on the x-axis
        ax.xaxis.set_major_locator(MaxNLocator(nbins=25))
        plt.ylabel('Pressure [hPa]')
        plt.title(f'Lat. Lon.: {latlon}')
        plt.tight_layout()
        fig.savefig(f'all-ozone-{saven}.jpg',dpi=300)
        return
    
    
    def check_one_case(case:str):
        """
        Summary: testing reading one MIPAS O3 data

        Args:
            case (str): path of testing case
        Return:
            None
        Output:
            JPG files named "one-ozone-{p/h}.jpg"
        """
        # testing with 1 case first
        dt = read_netCDF4(file_path = case)
        pressure = get_attribute(att_name='pressure',dt=dt)
        ozone = get_attribute(att_name='target',dt=dt)
        altitude = get_attribute(att_name='altitude',dt=dt)
        TKY_lat_idx,TKY_lon_idx,lat,lon = getTKY_idx(dataset=dt)
        check_idx_lat_lon(lat=TKY_lat_idx,lon=TKY_lon_idx)
        
        id = np.copy(TKY_lat_idx) # lat or lon 
        TKY_ozone = ozone[:,id]  
        TKY_pressure = pressure[:,id]
        TKY_altitude = altitude[:,id]
        # check stats
        check_stats(values=TKY_ozone)
        check_stats(values=TKY_pressure)
        # plot ozone: p && h
        plot_O3(p=TKY_pressure,h=None,
            dt=TKY_ozone,latlon=(lat,lon))
        plot_O3(p=None,h=TKY_altitude,
            dt=TKY_ozone,latlon=(lat,lon))
        return

    def list_netCDF4_files(folder_path: str = './data'):
        """
        Summary: list up all netCDF4 files in path

        Args:
            folder_path (str, optional): Defaults to './data'.

        Raises:
            FileNotFoundError: warning if no file found

        Returns:
            nc_files (list): list of nc files in the path
        """
        # Check if the folder path exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The specified folder '{folder_path}' does not exist.")

        # Get a list of .nc files in the folder
        nc_files = glob.glob(os.path.join(folder_path, '*.nc'))
        return nc_files
        
    def loop_all_cases(doplot: bool = True):
        """
        Summary: main dish, read all O3 data and plot

        Args:
            doplot (bool, optional): flag to control plotting. Defaults to True.
        
        Output:
            JPGs from plot_all_O3 function

        Returns:
            o3_mtx (np.array): matrix of O3 values
            p_mtx (np.array): matrix of pressure levels
            h_mtx (np.array): matrix of altitude
            (lat,lon) (tuple): values of latitude and longitude
        """
        fns = list_netCDF4_files()
        o3_mtx = []
        p_mtx = []
        h_mtx = []
        for i,fn in enumerate(fns):
            dt = read_netCDF4(file_path=fn)
            pressure = get_attribute(att_name='pressure',dt=dt)
            ozone = get_attribute(att_name='target',dt=dt)
            altitude = get_attribute(att_name='altitude',dt=dt)
            TKY_lat_idx,TKY_lon_idx,lat,lon = getTKY_idx(dataset=dt)
            check_idx_lat_lon(lat=TKY_lat_idx,lon=TKY_lon_idx)
            id = np.copy(TKY_lat_idx) # lat or lon 
            TKY_ozone = ozone[:,id]  
            TKY_pressure = pressure[:,id]
            TKY_altitude = altitude[:,id]
            # check stats
            check_stats(values=TKY_ozone)
            check_stats(values=TKY_pressure)
            o3_mtx.append(np.array(TKY_ozone))
            p_mtx.append(np.array(TKY_pressure))
            h_mtx.append(np.array(TKY_altitude))
        o3_mtx = np.array(o3_mtx)
        p_mtx = np.array(p_mtx)
        h_mtx = np.array(h_mtx)
        # plot matrix
        if doplot:
            plot_all_O3(dt=o3_mtx,latlon=(lat,lon),fns=fns,
                    p=p_mtx,h=None)
            plot_all_O3(dt=o3_mtx,latlon=(lat,lon),fns=fns,
                    p=None,h=h_mtx)
        return o3_mtx, p_mtx, h_mtx, (lat,lon)

    def interpolate_O3(o3: np.array, p: np.array, newp: np.array = None):
        """
        Summary: to interpolate new O3 data by new pressure levels

        Args:
            o3 (np.array): original O3 values
            p (np.array): original pressure levels
            newp (np.array, optional): New pressure levels. Defaults to None.

        Returns:
            newp (np.array): new pressure levels
            len_newp (int): length of new pressure levels 
            interpolated_data (np.array): interpolated O3 data
        """
        if newp is None:
            newp = np.loadtxt('./TAPE5/TAPE5-p.txt').ravel()
        len_newp = len(newp)
        # Create an interpolation function using cubic interpolation
        interp_func = interp1d(p, o3, kind='cubic', fill_value='extrapolate')
        # Interpolate the data to the target levels
        interpolated_data = interp_func(newp)
        #print(interpolated_data)
        return newp,len_newp,interpolated_data
        
    def compute_mean_below_60km(
            o3_mtx: np.array, p_mtx: np.array,
            h_mtx: np.array, latlon: tuple):
        """
        Summary: calculate mean O3 values below 60 km

        Args:
            o3_mtx (np.array): matrix of all O3 values
            p_mtx (np.array): matrix of all pressure levels
            h_mtx (np.array): matrix of all altitudes
            latlon (tuple): values of latitude and longitude

        Returns:
            newp (np.array): unified pressure levels according to 1 data of p_mtx
            mean_newo3 (np.array): mean O3 values after interpolating to newp
        """
        print(f"Shape of o3_mtx: {o3_mtx.shape}")
        print(f"Shape of p_mtx: {p_mtx.shape}")
        print(f"Shape of h_mtx: {h_mtx.shape}")
        # interpolate and adjust to newp
        # initialize
        newp,len_newp,newo3 = interpolate_O3(o3=o3_mtx[0,:],p=p_mtx[0,:])
        print(f'New {len_newp} pressure levels (TAPE5.base): {newp}')
        # loop for the rest and stack
        #newo3, newp, newh 
        for i,row in enumerate(o3_mtx[1:,:]):
            newp,len_newp,tmpo3 = interpolate_O3(o3=row,p=p_mtx[1+i,:])
            newo3 = np.vstack((newo3,tmpo3))
        # take mean & std along the y-axis
        print(f"New shape of o3_mtx: {newo3.shape}")
        mean_newo3 = np.mean(newo3,axis=0)
        print(f"Shape of mean_O3: {mean_newo3.shape}")
        std_newo3 = np.std(newo3,axis=0)
        print(f"Shape of std_O3: {std_newo3.shape}")
        # save std -> note: it is in TP5 model p levels !
        np.savetxt('std_newo3.txt',std_newo3)
        print(f"Std_newo3 is saved.")
        plot_vertical_mean_O3(dt=newo3,meandt=mean_newo3,meandtp=newp,std_newo3=std_newo3,
               latlon=latlon,p=newp,
           svn='mean-v')
        plot_heatmap_mean_O3(dt=newo3,meandt=mean_newo3,meandtp=newp,std_newo3=std_newo3,
               latlon=latlon,p=newp,
           svn='mean-hm')
        return newp, mean_newo3
    
    def read_TAPE5(fn: str):
        """
        Summary: read TAPE5 (configuration file of LBLRTM)

        Args:
            fn (str): file name of TAPE5

        Returns:
            dt: read TAPE5 row by row (only read model atmosphere part)
        """
        #dt = np.loadtxt(fn,dtype=str)
        with open(fn,'r') as file:
            dt = file.readlines()
        #dt = np.array(dt)
        #print(dt[21])
        #print(dt[:][22:31])
        return dt
    
    def modify_TAPE5(TP5_model_atm: list, mean_newo3: np.array, range: tuple):
        """
        Summary: modify TAPE5 using mean O3 values

        Args:
            TP5_model_atm (list): TAPE5 model atmosphere read by open statement
            mean_newo3 (np.array): mean O3 values unified to same pressure levels
            range (tuple): range of columns in TAPE5 that are O3 values (see LBLRTM documentation for TAPE5)

        Returns:
            TP5_model_atm (list): new TAPE5 with updated O3 values
        """
        s,e = range  # s and e stands for start and end of the cols of record
        n=0
        # note: o3 MIPAS in VMR*1e-6 unit, so e.q. to PPM
        for i,row in enumerate(TP5_model_atm):
            if (i%2 == 0 and row[0] != "%"):
                # replace '2' default unit to 'A' PPM unit
                new_row = row.replace(TP5_model_atm[i][40:47], "2AA22A2")
                #print(new_row)
                TP5_model_atm[i] = new_row
            elif (i%2 != 0 and row[0] != "%"):
                # format mean_newo3 to string && 14 length with 8 digist after decimal
                new_row = row.replace(TP5_model_atm[i][s:e], f"{mean_newo3[n]:14.8E}")
                #print(row)
                #print(new_row)
                TP5_model_atm[i] = new_row
                n += 1
            else:
                continue
        return TP5_model_atm
    
    def write_base_TAPE5(mean_newo3: np.array, fn_base_TP5: str):
        """
        Summary: read base TAPE5 and update with new O3 values

        Args:
            mean_newo3 (np.array): mean O3 values from MIPAS O3 data
            fn_base_TP5 (str): file name of base TAPE5 file with root path

        Returns:
            new_TP5_model_atm (list): updated TAPE5 model atmosphere
        """
        #TP5_atm_base = read_TAPE5(fn='./TAPE5/TAPE5-atm-base.txt')
        TP5_atm_base = read_TAPE5(fn=fn_base_TP5)
        # for o3, range is 32:46
        o3_range = (31,45)
        new_TP5_model_atm = modify_TAPE5(TP5_model_atm=TP5_atm_base, mean_newo3=mean_newo3, range=o3_range) 
        #print(new_TP5_model_atm)
        return new_TP5_model_atm
    
    def save_modified_TAPE5(new_TP5_model_atm: list, fn: str):
        """
        Summary: store updated TAPE5 model atmosphere

        Args:
            new_TP5_model_atm (list): updated TAPE5
            fn (str): file name
        """
        with open(fn,'w') as file:
            file.writelines(new_TP5_model_atm)
        return
    
    def check_TAPE5(TP5: list):
        """To check if TAPE5 is None"""
        if TP5 is None:
            print('!!!!!!!!!!!It is None!!!!!!!')
        return
    
    def save_new_O3(p: np.array, o3: np.array, fn: str):
        """
        Summary: save mean O3 values to txt for further process (i.e., as a priori)

        Args:
            p (np.array): array of pressure levels (unified)
            o3 (np.array): array of mean O3 values (interpolated to same level before mean)
            fn (str): file name
        """
        path = "./apriori/"
        dt = np.column_stack((p,o3))
        np.savetxt(path+fn,dt)
        print(f"{fn} created.")
        return
                
    ## Start [check 1 case]:
    check_one_case(case='./data/MIPAS-E_IMK.201107.V8R_O3_561.nc')
    ## End [check 1 case]
    ## Start [read all, compute mean, and update TAPE5]
    o3_mtx, p_mtx, h_mtx, (lat,lon) = loop_all_cases() 
    # compute mean: interpolate data to TAPE5.base pressure levels 
    newp,mean_newo3 = compute_mean_below_60km(
        o3_mtx=o3_mtx, p_mtx=p_mtx,
        h_mtx=h_mtx, latlon=(lat,lon))
    # update and output a new TP5 -> with MIPAS O3 profile (!exl. JRA-55 levels)
    newTP5_atm = write_base_TAPE5(mean_newo3=mean_newo3, 
                                  fn_base_TP5='./TAPE5/TAPE5-atm-base.txt')
    check_TAPE5(newTP5_atm)
    save_modified_TAPE5(new_TP5_model_atm=newTP5_atm,
                        fn='./TAPE5/TAPE5-atm-MIPAS.txt')
    # save final p-lv and newo3
    save_new_O3(p=newp, o3=mean_newo3, fn='MIPAS-O3-MLS-apriori.dat')
    # process and save JRA-55 GPV levels: total 22 layers
    p_jra55 = np.array([1000,975,950,925,900,
                        850,800,700,600,500,
                        400,300,250,200,150,
                        100,70,50,30,20,10,
                        1,0.1])
    p_jra55,len_p,mean_jra55o3 = interpolate_O3(o3=mean_newo3,
                                                p=newp, newp=p_jra55)
    save_new_O3(p=p_jra55,o3=mean_jra55o3,fn='MIPAS-O3-GPV-apriori.dat')
    ## End [read all, compute mean, and update TAPE5]


if __name__=="__main__":
    main()
# end