
import matplotlib.pyplot as plt
plt.style.use(['classic'])
import numpy as np
import astropy.coordinates as coord
import pandas as pd
from scipy.interpolate import interp1d

plt.rcParams['font.family']='stixgeneral'
plt.rcParams.update({'font.size':16})

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u1
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from itertools import groupby
import scipy.integrate as integrate
import scipy.special as special
from scipy.stats import poisson

satidflag = np.loadtxt('halo12_bound.txt')
#flag2 = np.loadtxt('halo17_bound_circ.txt')


def make_csv(data_name, deg, num, particle):  # string, float (rotation angle in degrees)
    import ebf
    import pandas as pd
    import numpy as np

    rot = deg * np.pi / 180.  # angle in radians

    data = ebf.read(data_name + '.ebf')  # reads in ebf file for galaxia output
    df = pd.DataFrame()  # creates empty pandas DataFrame

    keys = data.keys()  # creates list of keys for the ebf file

    if particle == True:
        cut = (data['partid'] == 0)
    else:
        cut = ()
    flag = np.unique(data['satid'])[np.where(satidflag == 0)]
    # print (keys)
    # for ii in range(1,len(list(data.keys()))): #for each list in the ebf file, create a dataframe column with that key as a column name
    # s=pd.Series(data[name])
    # print (s)

    # df[list(data.keys())[ii]]=data[list(data.keys())[ii]]
    # print (s)

    index = range(0, len(data['smass'][cut][np.isin(data['satid'][cut], flag)]), 1)
    indexi = np.random.choice(index, num, replace=False)

    df['smass'] = data['smass'][cut][np.isin(data['satid'][cut], flag)][indexi]

    df['age'] = data['age'][cut][np.isin(data['satid'][cut], flag)][indexi]
    df['rad'] = data['rad'][cut][np.isin(data['satid'][cut], flag)][indexi]

    df['mag0'] = data['mag0'][cut][np.isin(data['satid'][cut], flag)][indexi]
    df['mag1'] = data['mag1'][cut][np.isin(data['satid'][cut], flag)][indexi]
    df['mag2'] = data['mag2'][cut][np.isin(data['satid'][cut], flag)][indexi]
    df['popid'] = data['popid'][cut][np.isin(data['satid'][cut], flag)][indexi]
    df['satid'] = data['satid'][cut][np.isin(data['satid'][cut], flag)][indexi]
    # df['fieldid']=data['fieldid']
    # df['partid']=data['partid']
    # df['center']=data['center']
    df['lum'] = data['lum'][cut][np.isin(data['satid'][cut], flag)][indexi]
    df['teff'] = data['teff'][cut][np.isin(data['satid'][cut], flag)][indexi]

    # set original output positions and velocities as xx_0. These are in the unrotated frame where x-axis does not always point to galactic center
    df['glon_0'] = data['glon'][cut][np.isin(data['satid'][cut], flag)][indexi]
    df['glat_0'] = data['glat'][cut][np.isin(data['satid'][cut], flag)][indexi]
    df['px_0'] = data['px'][cut][np.isin(data['satid'][cut], flag)][indexi]
    df['py_0'] = data['py'][cut][np.isin(data['satid'][cut], flag)][indexi]
    df['pz_0'] = data['pz'][cut][np.isin(data['satid'][cut], flag)][indexi]
    df['vx_0'] = data['vx'][cut][np.isin(data['satid'][cut], flag)][indexi]
    df['vy_0'] = data['vy'][cut][np.isin(data['satid'][cut], flag)][indexi]
    df['vz_0'] = data['vz'][cut][np.isin(data['satid'][cut], flag)][indexi]

    # print (df['glon_0'],df['glon'])

    # set outputs by rotating so x-axis points to galactic center
    df['glon'] = (df['glon_0'] - deg) % 360
    df['glat'] = df['glat_0']
    df['px'] = df['px_0'] * np.cos(rot) + df['py_0'] * np.sin(rot)
    df['py'] = -df['px_0'] * np.sin(rot) + df['py_0'] * np.cos(rot)
    df['pz'] = df['pz_0']
    df['vx'] = df['vx_0'] * np.cos(rot) + df['vy_0'] * np.sin(rot)
    df['vy'] = -df['vx_0'] * np.sin(rot) + df['vy_0'] * np.cos(rot)
    df['vz'] = df['vz_0']

    # set original center values
    o_cen = np.full(len(df['glon']), np.nan)

    o_cen[0:6] = [-8.0, 0.0, 0.015, 11.1, 239.08, 7.25]
    df['orig_center'] = o_cen

    df.to_csv(data_name + 'bound.csv', index=False)  # save as CSV with same base name that can be opened in python 3
    return df

testd = make_csv('halo12', 0,60000,False)

def get_coord_obj_rot(data):  # pandas DataFrame
    import astropy
    import astropy.coordinates as coord
    import astropy.units as u
    import numpy as np
    x = data['px'].values  # full sets in heliocentric coords
    y = data['py'].values
    z = data['pz'].values
    vx = data['vx'].values
    vy = data['vy'].values
    vz = data['vz'].values

    v_sun_gc = [data['orig_center'][3], data['orig_center'][4], data['orig_center'][
        5]]  # galactocentric sun coordinates, based on original heliocentric x-axis aligned with galactocentric X-axis

    gal = coord.SkyCoord(u=x * u.kpc, v=y * u.kpc, w=z * u.kpc,  # galactic frame cartesian coordinate object
                         U=vx * u.km / u.s, V=vy * u.km / u.s, W=vz * u.km / u.s, frame='galactic',
                         representation_type='cartesian', differential_type='cartesian')
    gal.representation_type = 'spherical'  # change type to spherical (l, b, proper motions, rad vel, etc)
    gal.differential_type = coord.representation.SphericalCosLatDifferential

    return gal  # return coordinate object


# TURNS A DATAFRAME OF STARS INTO ASTROPY COORDINATE OBJECT, GALACTIC FRAME (HELIOCENTRIC X-AXIS TO GC), TAKING INTO ACCOUNT THE SOLAR REFLEX MOTION
def get_coord_obj_rot_gsr(data):
    import astropy
    import astropy.coordinates as coord
    import astropy.units as u
    import numpy as np
    x = data['px'].values  # full sets in heliocentric coords
    y = data['py'].values
    z = data['pz'].values
    vx = data['vx'].values
    vy = data['vy'].values
    vz = data['vz'].values
    # v_sun_gl=[data['center'][3], data['center'][4], data['center'][5]]
    v_sun_gc = [11.1, 239.08, 7.25]  # galactocentric sun coordinates
    v_sun = coord.CartesianDifferential(v_sun_gc * u.km / u.s)  # turn into a coordinate cartesian differential

    gal = coord.SkyCoord(u=x * u.kpc, v=y * u.kpc, w=z * u.kpc,
                         U=(vx + v_sun_gc[0]) * u.km / u.s, V=(vy + v_sun_gc[1]) * u.km / u.s,
                         W=(vz + v_sun_gc[0]) * u.km / u.s, frame='galactic',
                         representation_type='cartesian',
                         differential_type='cartesian')  # adds solar velocity to star velocity for calculating gsr frame parameters
    print(gal)
    gal.representation_type = 'spherical'  # convert to spherical representation (l, b, distance)

    gal.differential_type = coord.representation.SphericalCosLatDifferential  # convert differentials to spherical cos lat (proper motions, radial velocity)

    return gal


# takes an angle and returns corresponding angle within limits. If no minparam give, returns angle between (0,2*pi)/(0,360)
# if minval is given returns angle between (minval, minval+2*pi)/(minval, minval+360). Ex if minval=-pi, range will be (-pi, pi)
def limrange(angle, *minparam, **degflag):  # float, float, bool (deg=True/False)
    import numpy as np
    if len(minparam) == 0:
        minval = 0.0
    else:
        minval = minparam[0]

    if ('deg' in degflag):
        newangle = (angle - minval) % (360.0) + minval
    else:
        newangle = (angle - minval) % (2.0 * np.pi) + minval

    return newangle


# add radial velocity, l, b, dist, proper motions, v_gsr, pm_gsr, v_t, v_t_gsr, dist modulus to CSV
def rv_csv(halo, field, deg):  # string, string, int
    import pandas as pd
    import numpy as np
    # NAME OF CSV FILE
    if deg % 1 == 0:  # if deg is integer use make string
        d = str(int(deg))
    else:
        d = "{0:.2f}".format(deg)  # else round to 2 decimals and make string
    data_name = halo + '/' + d + '/' + field + '/' + halo + '_' + field + '_' + d
    # OPEN CSV
    data = pd.read_csv(data_name + '.csv')
    # GALACTOCENTRIC SOLAR VELOCITY WITH SUN ON X-AXIS
    v_sun_gc = [11.1, 239.08, 7.25]
    # CREATE ASTROPY COORD OBJECT
    gal = get_coord_obj_rot(data)
    gal_gsr = get_coord_obj_rot_gsr(data)
    # ASSIGN TO DATAFRAME
    data = data.assign(l_coord=gal.l.degree)
    data = data.assign(b_coord=gal.b.degree)
    data = data.assign(pm_l_cosb=gal.pm_l_cosb)
    data = data.assign(pm_b=gal.pm_b)
    data = data.assign(pm_mag=np.sqrt(np.add(np.square(data.pm_b), np.square(data.pm_l_cosb))))
    data = data.assign(r_v=gal.radial_velocity)
    data = data.assign(dist=gal.distance)
    data = data.assign(dm=5. * np.log10(gal.distance.value * 1000.) - 5.)
    data = data.assign(v_l=4.74047 * gal.pm_l_cosb * gal.distance)
    data = data.assign(v_b=4.74047 * gal.pm_b * gal.distance)
    data = data.assign(v_t=np.sqrt(np.add(np.square(data.v_l), np.square(data.v_b))))
    data = data.assign(pm_l_cosb_gsr=gal_gsr.pm_l_cosb)
    data = data.assign(pm_b_gsr=gal_gsr.pm_b)
    data = data.assign(pm_mag_gsr=np.sqrt(np.add(np.square(data.pm_b_gsr), np.square(data.pm_l_cosb_gsr))))
    data = data.assign(v_gsr=gal_gsr.radial_velocity)
    data = data.assign(v_l_gsr=4.74047 * gal_gsr.pm_l_cosb * gal_gsr.distance)
    data = data.assign(v_b_gsr=4.74047 * gal_gsr.pm_b * gal_gsr.distance)
    data = data.assign(v_t_gsr=np.sqrt(np.add(np.square(data.v_l_gsr), np.square(data.v_b_gsr))))
    # SAVE NEW CSV WITH SAME NAME TO REPLACE
    data.to_csv(data_name + '.csv', index=False)


rad_t = get_coord_obj_rot_gsr(testd)
gc1 = rad_t.transform_to(coord.Galactocentric)
r2= np.sqrt(gc1.x**2+gc1.y**2+gc1.z**2)
print ('   1',rad_t)


# Function for calculating the space separation and the velocity difference in each shell



zmin = 20
zmax = 100
zlim = 4.

ra = r2/u1.kpc
r2= ra

vral = rad_t.radial_velocity/u1.km*u1.s
groupidi = range(0,len(testd))


# cutting r and z > 20 & r < 60 kpc stars
df1=pd.DataFrame()
df2=pd.DataFrame()
df3 = pd.DataFrame()
df1['x']=gc1.x[ (zmin < r2 ) & (r2 < zmax ) ]
df1['y']=gc1.y[ (zmin < r2 ) & (r2< zmax ) ]
df1['z']=gc1.z[ (zmin < r2 ) & (r2 < zmax ) ]
df1['Vgal']= vral[ (zmin <r2 ) & (r2 < zmax ) ]
df1['r']= r2[ (zmin < r2) & (r2 < zmax )]
df1['starid']= np.array(groupidi)[ (zmin < r2) & (r2 < zmax )]
df1['index']= np.arange(0,len(df1['r']),1)
#print (df1['index'])
df2['r']=df1['r'][(np.absolute(df1['z']) > zlim)]
df2['x']=df1['x'][(np.absolute(df1['z']) > zlim)]
df2['y']=df1['y'][(np.absolute(df1['z']) > zlim) ]
df2['z']=df1['z'][(np.absolute(df1['z']) > zlim)]
df2['Vgal']= df1['Vgal'][(np.absolute(df1['z']) > zlim)]
df2['starid']= df1['starid'][(np.absolute(df1['z']) > zlim)]
df2['index']= np.arange(0,len(df2['r']),1)


df3 = df2.sort_values(by=['index'])

df3.to_csv("df3_60000_halo12.csv")

# class for the input of the correlation function
class Comp:
    def __init__(self, x, y, z, rv, groupid):
        '''
        class for the input of the correlation function
        :param x: x position for the target
        :param y: y position for the target
        :param z: z position for the target
        :param rv: radial velocity for the target

        '''

        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.rv = np.array(rv)
        self.groupid = np.array(groupid)


# Function for calculating the space separation and the velocity difference in each shell

def v_diff(spaces, data, interval):
    '''
        :param spaces: Scale for space sparation
        :param data: The orginal dataset
        :param interval: raidal space interval for the shell of calculating the space separation


    '''

    vdiffh = []
    rlim = []
    num = []
    spect = []
    indexout = []
    n = 0
    for ii in range(int(20), int(70), interval):

        rlim.append(ii)
        zmin = ii
        zmax = ii + interval
        dfx = pd.DataFrame()
        dfxf = pd.DataFrame()
        dfx['x'] = data['x'][(zmin < data['r']) & (data['r'] < zmax)]
        # print (dfx['x'])
        dfx['y'] = data['y'][(zmin < data['r']) & (data['r'] < zmax)]
        dfx['z'] = data['z'][(zmin < data['r']) & (data['r'] < zmax)]
        dfx['Vgal'] = data['Vgal'][(zmin < data['r']) & (data['r'] < zmax)]
        dfx['index'] = data['index'][(zmin < data['r']) & (data['r'] < zmax)]

        dfx['r'] = data['r'][(zmin < data['r']) & (data['r'] < zmax)]
        indexf = []
        vdiff = []
        spacedt = []
        num.append(len(dfx['r']))
        dfx2 = dfx.sort_values(by=['index'])

        indexp = [x for x in data.index if x not in dfx2.index]
        indexs = np.append(np.array(dfx2.index), np.array(indexp))

        dfxf = data.loc[indexs]

        # print (dfx2,dfxf)

        # print (dfx2['r'])
        k = Comp(dfx2['x'], dfx2['y'], dfx2['z'], dfx2['Vgal'], dfx2['index'])
        k2 = Comp(dfxf['x'], dfxf['y'], dfxf['z'], dfxf['Vgal'], dfxf['index'])

        # print (k.rv,k2.rv)
        # vsort = k.rv[np.argsort(dfx['r'])]
        # xsort = k.x[np.argsort(dfx['r'])]
        # ysort = k.y[np.argsort(dfx['r'])]
        # zsort = k.z[np.argsort(dfx['r'])]
        # print ('v1',k.rv,k.x)
        # print ('v2',k2.rv,k2.x)



        # calculating the velocity difference and space separation for each shell
        for jj in range(0, len(k.rv)):

            # print (k.rv,k2.rv)


            vi = np.repeat(k.rv[jj], len(k2.rv))
            xi = np.repeat(k.x[jj], len(k2.rv))
            yi = np.repeat(k.y[jj], len(k2.rv))
            zi = np.repeat(k.z[jj], len(k2.rv))

            diffv = (vi - k2.rv) ** 2
            diffx = (xi - k2.x) ** 2
            diffy = (yi - k2.y) ** 2
            diffz = (zi - k2.z) ** 2
            indext = []
            for kk in range(0, len(k2.groupid)):
                indext.append(str(k.groupid[jj]) + '-' + str(k2.groupid[kk]))

            # diff = diffv+diffx+diffy+diffz
            spaced = diffx + diffy + diffz

            indexf.append(indext)
            # dia.append(diff)
            vdiff.append(vi - k2.rv)
            spacedt.append(np.sqrt(spaced))
        # print (len(vdiff))




        # diaf = np.array(dia)[np.triu_indices(len(dfx['r']),1)]
        vdiffto = np.array(vdiff)[np.triu_indices(len(k.rv), 1)]
        specdto = np.array(spacedt)[np.triu_indices(len(k.rv), 1)]
        indexfo = np.array(indexf)[np.triu_indices(len(k.rv), 1)]
        spect.append(specdto)
        indexout.append(indexfo[specdto < spaces])

        # print (len(vdiffto),len(specdto))

        vdiffh.append(vdiffto[specdto < spaces])

        n = n + 1

    return rlim, vdiffh, num, spect, indexout


rlimf = []
test1f = []
numf = []
sepf = []
indexf = []

for scale in range(2, 8, 2):
    rlim, test1, num, sep, indexii = v_diff(scale, df3, 8)
    rlimf.append(rlim)
    test1f.append(test1)
    numf.append(num)
    sepf.append(sep)
    indexf.append(indexii)











#shell scale = 10 kpc range 2-12 kpc


dff = pd.DataFrame()
for sc in range(0, len(test1f)):
    for k in range(0, len(test1f[1])):
        hist, bin_edges = np.histogram(test1f[sc][k], density=False, bins=100)

        dff[str(sc)+str(k)+' '+str(int(rlimf[sc][k])) + '-' + str(int(rlimf[sc][k] + 8)) + 'kpc x'] = hist
        dff[str(sc)+str(k)+' '+str((int(rlimf[sc][k]))) + '-' + str(int(rlimf[sc][k] + 8)) + 'kpc y'] = bin_edges[:100]

dff.to_csv("halo12_veldiff_test.csv")








