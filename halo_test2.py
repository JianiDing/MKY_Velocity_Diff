
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
from scipy import stats
from scipy.stats import binned_statistic
from scipy.linalg import block_diag,eigh
from astropy.io import fits
import ebf


def make_csv(data_name, deg): #string, float (rotation angle in degrees)
    import ebf
    import pandas as pd
    import numpy as np


    rot=deg*np.pi/180. #angle in radians

    data=ebf.read(data_name+'.ebf') #reads in ebf file for galaxia output
    df=pd.DataFrame() #creates empty pandas DataFrame
    keys=data.keys() #creates list of keys for the ebf file
    #    print keys
    for name in keys: #for each list in the ebf file, create a dataframe column with that key as a column name
        s=pd.Series(data[name])
        df[name]=s

    #set original output positions and velocities as xx_0. These are in the unrotated frame where x-axis does not always point to galactic center
    df['glon_0']=df['glon']
    df['glat_0']=df['glat']
    df['px_0']=df['px']
    df['py_0']=df['py']
    df['pz_0']=df['pz']
    df['vx_0']=df['vx']
    df['vy_0']=df['vy']
    df['vz_0']=df['vz']

    #set outputs by rotating so x-axis points to galactic center
    df['glon']=(df['l_0']-deg)%360
    df['glat']=df['b_0']
    df['px']=df['x_0']*np.cos(rot)+df['y_0']*np.sin(rot)
    df['py']=-df['x_0']*np.sin(rot)+df['y_0']*np.cos(rot)
    df['pz']=df['z_0']
    df['vx']=df['vx_0']*np.cos(rot)+df['vy_0']*np.sin(rot)
    df['vy']=-df['vx_0']*np.sin(rot)+df['vy_0']*np.cos(rot)
    df['vz']=df['vz_0']

    #set original center values
    o_cen=np.full(len(df.l), np.nan)
    o_cen[0:6]=[-8.0, 0.0, 0.015, 11.1, 239.08, 7.25]
    df['orig_center']=o_cen

    df.to_csv(data_name+'.csv', index=False) #save as CSV with same base name that can be opened in python 3
#    return df


test = make_csv('halo02_all_sky_rot', 90)
