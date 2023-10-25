import matplotlib.pyplot as plt
plt.style.use(['classic'])
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from astropy.io import fits
from scipy import stats
from scipy.stats import binned_statistic
from scipy.linalg import block_diag,eigh
plt.rcParams['font.family']='stixgeneral'
plt.rcParams.update({'font.size':16})
from astropy.io import fits
import pdb
from mpl_toolkits.mplot3d import Axes3D
from astropy.coordinates import (CartesianRepresentation,CartesianDifferential)
from astropy.coordinates import Galactic
from astropy import table
from astropy.coordinates import SkyCoord
import csv
import pandas
from astropy.table import Table, Column, MaskedColumn

import astropy.io.fits as fits
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits as pyfits
import pandas as pd
import scipy.optimize as optim


ironrv = t1 = table.Table(pyfits.open('/Users/Jenny/thesis_idea_test/MKY_draco/rvtab-hpxcoadd-all.fits')[1].data)
t1_fiber = table.Table(pyfits.open('/Users/Jenny/thesis_idea_test/MKY_draco/rvtab-hpxcoadd-all.fits')[2].data)
t4 = table.Table(pyfits.open('/Users/Jenny/thesis_idea_test/MKY_draco/rvtab-hpxcoadd-all.fits')[4].data)

t1_comb = table.hstack((t1,t1_fiber,t4))

print('# before unique selection:', len(t1_comb))

# do a unique selection based on TARGET ID. Keep the first one for duplicates
# (and first one has the smallest RV error)
t1_unique = table.unique(t1_comb, keys='TARGETID_1', keep='first')
print('# after unique selection:', len(t1_unique))
testiron=t1_unique

def rv_to_gsr(c, v_sun=None):
    """Transform a barycentric radial velocity to the Galactic Standard of Rest
    (GSR).

    The input radial velocity must be passed in as a

    Parameters
    ----------
    c : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
        The radial velocity, associated with a sky coordinates, to be
        transformed.
    v_sun : `~astropy.units.Quantity`, optional
        The 3D velocity of the solar system barycenter in the GSR frame.
        Defaults to the same solar motion as in the
        `~astropy.coordinates.Galactocentric` frame.

    Returns
    -------
    v_gsr : `~astropy.units.Quantity`
        The input radial velocity transformed to a GSR frame.

    """
    if v_sun is None:
        v_sun = coord.Galactocentric().galcen_v_sun.to_cartesian()

    gal = c.transform_to(coord.Galactic)
    cart_data = gal.data.to_cartesian()
    unit_vector = cart_data / cart_data.norm()

    v_proj = v_sun.dot(unit_vector)

    return c.radial_velocity + v_proj

coord.galactocentric_frame_defaults.set('latest')


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


def v_diff(spaces, data, interval, times):
    '''
        :param spaces: Scale for space sparation
        :param data: The orginal dataset
        :param interval: raidal space interval for the shell of calculating the space separation

        :return vdiff: The mean velocity difference for each shell

    '''


    #vdiffhf: The mean velocity difference for each shell
    vdiffhf = []
    rlim = []
    num = []
    spectf = []
    indexoutf = []

    zmini = [20, 26, 32, 38, 45, 56, 65]
    zmaxi = [26, 32, 38, 45, 56, 65, 130]
    frac = []
    for ii in range(0, 7, 1):

        rlim.append(zmini[ii])
        zmin = zmini[ii]
        zmax = zmaxi[ii]
        dfx = pd.DataFrame()
        dfxf = pd.DataFrame()
        dfx['x'] = data['x'][(zmin < data['r']) & (data['r'] < zmax)]
        # print (dfx['x'])
        dfx['y'] = data['y'][(zmin < data['r']) & (data['r'] < zmax)]
        dfx['z'] = data['z'][(zmin < data['r']) & (data['r'] < zmax)]
        dfx['Vgal'] = data['Vgal'][(zmin < data['r']) & (data['r'] < zmax)]
        dfx['index'] = data['index'][(zmin < data['r']) & (data['r'] < zmax)]

        dfx['r'] = data['r'][(zmin < data['r']) & (data['r'] < zmax)]
        # zmin=zmax
        indexf = []
        vdiff = []
        spacei = []
        num.append(len(dfx['r']))
        # time1 = data['time'][(zmin < data['r']) & (data['r'] < zmax)]
        # frac1= len(time1[time1 < 8.0])/len(time1)
        ## frac3= len(time1[time1 < 12.0])/len(time1)
        # frac.append([frac1,frac2,frac3])

        for kk in range(0, len(dfx), times):
            minn = kk
            maxx = kk + times
            # print (minn,maxx)

            v, s = pair_cal(data, dfx[minn:maxx], spaces)
            # print (len(dfx))

            # indexf.append(i)
            vdiff.append(v)
            spacei.append(s)
            # print (v)
        # print (vdiff)
        vdiffhf.append(np.concatenate((vdiff)))
        # print (vdiffhf)
        spectf.append(np.concatenate(spacei))
        # indexoutf.append(np.concatenate(indexf))

    return rlim, vdiffhf, num, spectf, frac


def pair_cal(dfall, df, spaces):
    dfx2 = df.sort_values(by=['index'])

    indexp = [x for x in dfall.index if x not in dfx2.index]
    indexs = np.append(np.array(dfx2.index), np.array(indexp))

    dfxf = dfall.loc[indexs]

    k = Comp(dfx2['x'], dfx2['y'], dfx2['z'], dfx2['Vgal'], dfx2['index'])
    k2 = Comp(dfxf['x'], dfxf['y'], dfxf['z'], dfxf['Vgal'], dfxf['index'])

    # print (k.rv,k2.rv)

    indexf = []
    vdiff = []
    spacedt = []

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
        # for kk in range(0, len(k2.groupid)):
        # indext.append(str(k.groupid[jj]) + '-' + str(k2.groupid[kk]))

        # diff = diffv+diffx+diffy+diffz
        spaced = diffx + diffy + diffz

        # indexf.append(indext)
        # dia.append(diff)
        vdiff.append(vi - k2.rv)
        spacedt.append(np.sqrt(spaced))
        # diaf = np.array(dia)[np.triu_indices(len(dfx['r']),1)]
    vdiffto = np.array(vdiff)[np.triu_indices(len(k.rv), 1)]
    specdto = np.array(spacedt)[np.triu_indices(len(k.rv), 1)]
    # indexfo = np.array(indexf)[np.triu_indices(len(k.rv), 1)]
    # spect.append(specdto)
    # indexout.append(indexfo[specdto < spaces])


    return vdiffto[specdto < spaces], specdto





#MCMC fitting
from astropy.io import fits
import pdb
import emcee
import corner


def two_gnfunc(dataarr, params):
    '''
    function for model of the two gaussians
    :param dataarr: data array
    :param params: parameter for the two gaussian distributions (center (center for gaussian narrow gaussian)
    sigma (sigma for gaussian narrow gaussian), center2 = (center2 for gaussian narrow gaussian),sigma2 = (sigma2 for gaussian narrow gaussian)
    amp1 (fraction of area under narrow gaussian/area under broad gaussian))
    :return: normalizied two gaussian function
    '''

    center = params[0]
    sigma = params[1]
    center2 = params[2]
    sigma2 = params[3]
    Amp1 = params[4]

    # print(center, sigma, center2, sigma2, Amp1)

    gnvals = Amp1 * np.exp(-(dataarr - center) ** 2 / (2 * sigma * sigma)) / sigma / np.sqrt(2 * np.pi) \
             + (1 - Amp1) * np.exp(-(dataarr - center2) ** 2 / (2 * sigma2 * sigma2)) / sigma2 / np.sqrt(2 * np.pi)
    return gnvals


def twogau_like(params, xvals):
    # likelihood used in fitting (normalizied)
    modelvals = two_gnfunc(xvals, params)

    mlikelihood = - np.sum(np.log(modelvals))

    # print(mlikelihood)
    return (mlikelihood)


def log_prior(theta):
    m,b,log_f,c,k = theta
    if  -100 < m < 100 and 0 < b < 150 and -50 < log_f < 50 and 100 < c < 500 and 1e-5 < k < 1-1e-5:
        return 0.0
    return -np.inf

def log_probability(theta, x):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return -lp -  twogau_like(theta, x)


def data_collect(testiron, flag, path, ramin, ramax, decmin, decmax, scale):
    '''
    collect data for the fitting
    :param testiron: data including distance and velocity for the calculation
    :param flag: flag for the selection of the data
    :param path: path for the data
    :param ramin: l min
    :param ramax: l max
    :param decmin: b min
    :param decmax: b max
    :param scale: scale for the data
    :return: rlim (distance limit for the data), vlim (velocity limit for the data), vlim2 (velocity limit for the data), rlim2 (distance limit for the data)
    '''

    iqk, = np.where((testiron['RVS_WARN'] == 0) & (testiron['RR_SPECTYPE'] != 'QSO') & (testiron['VSINI'] < 50)

                    & (~np.isnan(testiron["PMRA_ERROR"])) & (~np.isnan(testiron["PMDEC_ERROR"])) & (
                        ~np.isnan(testiron["PMRA_PMDEC_CORR"])) & (testiron["LOGG"] < 6) & (testiron["LOGG"] > 3))

    imainblue, = np.nonzero(testiron['MWS_TARGET'][iqk] & flag)
    ms = testiron[iqk][imainblue]
    ms['dist'] = 1 / ms['PARALLAX_3']
    ramin = ramin * u.degree
    ramax = ramax * u.degree
    decmin = decmin * u.degree
    decmax = decmax * u.degree

    ranew = ms['RA'][(ms['dist'] > 20)]
    decnew = ms['DEC'][(ms['dist'] > 20)]
    dist = ms['dist'][(ms['dist'] > 20)]

    helio = SkyCoord(ra=ranew * u.degree, dec=decnew * u.degree, distance=dist * u.kpc)
    galactic2 = helio.galactic

    coord.galactocentric_frame_defaults.set('latest')
    t = pd.DataFrame()
    icrs = coord.SkyCoord(ra=ranew * u.deg, dec=decnew * u.deg,
                          radial_velocity=ms['VRAD'][(ms['dist'] > 20)] * u.km / u.s, frame='icrs')
    t['v'] = rv_to_gsr(icrs)
    gal = helio.transform_to(coord.Galactocentric)

    galc = gal.represent_as(coord.SphericalRepresentation)

    R = np.array(galc.distance.to_value(u.kpc))
    ms2 = pd.DataFrame()

    ms2['x'] = gal.cartesian.x
    ms2['y'] = gal.cartesian.y
    ms2['z'] = gal.cartesian.z
    ms2['r'] = R
    ms2['v'] = t['v']

    zmin = 20.
    zmax = 100
    zlim = 4.

    t = ms2

    # cutting r and z > 20 & r < 60 kpc stars
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df1['x'] = t['x'][(zmin < t['r']) & (t['r'] < zmax)]
    df1['y'] = t['y'][(zmin < t['r']) & (t['r'] < zmax)]
    df1['z'] = t['z'][(zmin < t['r']) & (t['r'] < zmax)]
    df1['Vgal'] = t['v'][(zmin < t['r']) & (t['r'] < zmax)]
    df1['r'] = t['r'][(zmin < t['r']) & (t['r'] < zmax)]
    df2['r'] = df1['r'][(np.absolute(df1['z']) > zlim)]
    df2['x'] = df1['x'][(np.absolute(df1['z']) > zlim)]
    df2['y'] = df1['y'][(np.absolute(df1['z']) > zlim)]
    df2['z'] = df1['z'][(np.absolute(df1['z']) > zlim)]
    df2['Vgal'] = df1['Vgal'][(np.absolute(df1['z']) > zlim)]

    # data for verifying Xue 2008 et al paper
    df3 = pd.DataFrame()

    df3['x'] = t['x'][(np.absolute(t['z']) > zlim)]
    df3['y'] = t['y'][(np.absolute(t['z']) > zlim)]
    df3['z'] = t['z'][(np.absolute(t['z']) > zlim)]
    df3['Vgal'] = t['v'][(np.absolute(t['z']) > zlim)]
    df3['r'] = t['r'][(np.absolute(t['z']) > zlim)]
    df3['index'] = np.arange(0, len(df3['r']), 1)
    # print (len(df3))
    rlimf = []
    test1f = []
    numf = []
    sepf = []
    indexf = []
    fract = []
    for scale in [scale]:
        rlim, test1, num, sep, frac = v_diff(scale, df3, 10, 5000, galactic2, ramin, ramax, decmin, decmax)

        rlimf.append(rlim)
        test1f.append(test1)
        numf.append(num)
        sepf.append(sep)
        # indexf.append(indexii)
        fract.append(frac)

    return rlimf, test1f



# t



def v_diff_patch_2(spaces, data, times):
    '''
        :param spaces: Scale for space sparation
        :param data: The orginal dataset
        :param interval: raidal space interval for the shell of calculating the space separation


    '''

    numf = []

    zmini = [20, 32, 44, 56]
    zmaxi = [32, 44, 56, 90]
    vdiffhff = []
    # print (vdiffhf)
    spectff = []
    indexoutff = []
    frac = []

    for ii in range(0, 4, 1):

        vdiffhf = []
        rlim = []
        spectf = []
        indexoutf = []
        num = []
        if ii < 3:
            rlim.append(zmini[ii])
            zmin = zmini[ii]
            zmax = zmaxi[ii]
            deg = math.degrees(math.atan(math.tan(20 / 20)))
            dfx = pd.DataFrame()
            dfxf = pd.DataFrame()

            dfx['x'] = data['x'][(zmin < data['r']) & (data['r'] < zmax)]
            # print (dfx['x'])
            dfx['y'] = data['y'][(zmin < data['r']) & (data['r'] < zmax)]
            dfx['z'] = data['z'][(zmin < data['r']) & (data['r'] < zmax)]
            dfx['Vgal'] = data['Vgal'][(zmin < data['r']) & (data['r'] < zmax)]
            dfx['index'] = data['index'][(zmin < data['r']) & (data['r'] < zmax)]
            dfx['theta'] = data['theta'][(zmin < data['r']) & (data['r'] < zmax)]
            dfx['phi'] = data['phi'][(zmin < data['r']) & (data['r'] < zmax)]

            dfx['r'] = data['r'][(zmin < data['r']) & (data['r'] < zmax)]
            degstheta = np.arange(0, 180, deg)
            degsphi = np.arange(-180, 180, deg)

            for jj in range(1, len(degstheta)):
                for kk in range(1, len(degsphi)):
                    dfxi = pd.DataFrame()
                    dfxi['Vgal'] = dfx['Vgal'][(degstheta[jj - 1] < dfx['theta']) & (dfx['theta'] < degstheta[jj]) & (
                                dfx['phi'] > degsphi[kk - 1]) & (dfx['phi'] < degsphi[kk])]
                    dfxi['index'] = dfx['index'][(degstheta[jj - 1] < dfx['theta']) & (dfx['theta'] < degstheta[jj]) & (
                                dfx['phi'] > degsphi[kk - 1]) & (dfx['phi'] < degsphi[kk])]
                    dfxi['x'] = dfx['x'][(degstheta[jj - 1] < dfx['theta']) & (dfx['theta'] < degstheta[jj]) & (
                                dfx['phi'] > degsphi[kk - 1]) & (dfx['phi'] < degsphi[kk])]
                    # print (dfx['x'])
                    dfxi['y'] = dfx['y'][(degstheta[jj - 1] < dfx['theta']) & (dfx['theta'] < degstheta[jj]) & (
                                dfx['phi'] > degsphi[kk - 1]) & (dfx['phi'] < degsphi[kk])]
                    dfxi['z'] = dfx['z'][(degstheta[jj - 1] < dfx['theta']) & (dfx['theta'] < degstheta[jj]) & (
                                dfx['phi'] > degsphi[kk - 1]) & (dfx['phi'] < degsphi[kk])]
                    # zmin=zmax
                    print(dfx['theta'][(degstheta[jj - 1] < dfx['theta']) & (dfx['theta'] < degstheta[jj]) & (
                                dfx['phi'] > degsphi[kk - 1]) & (dfx['phi'] < degsphi[kk])])

                    num.append(len(dfxi['x']))
                    # time1 = data['time'][(zmin < data['r']) & (data['r'] < zmax)]
                    # frac1= len(time1[time1 < 8.0])/len(time1)
                    # frac2= len(time1[time1 < 10.0])/len(time1)
                    # frac3= len(time1[time1 < 12.0])/len(time1)
                    # frac.append([frac1,frac2,frac3])
                    vdiffhf.append(dfxi['Vgal'].values)
                    # print (vdiffhf)
                    # spectf.append(np.concatenate(spacei))
                    indexoutf.append(dfxi['index'].values)
        else:

            rlim.append(zmini[ii])
            zmin = zmini[ii]
            zmax = zmaxi[ii]
            deg = math.degrees(math.atan(math.tan(25 / 20)))
            dfx = pd.DataFrame()
            dfxf = pd.DataFrame()

            dfx['x'] = data['x'][(zmin < data['r']) & (data['r'] < zmax)]
            # print (dfx['x'])
            dfx['y'] = data['y'][(zmin < data['r']) & (data['r'] < zmax)]
            dfx['z'] = data['z'][(zmin < data['r']) & (data['r'] < zmax)]
            dfx['Vgal'] = data['Vgal'][(zmin < data['r']) & (data['r'] < zmax)]
            dfx['index'] = data['index'][(zmin < data['r']) & (data['r'] < zmax)]
            dfx['theta'] = data['theta'][(zmin < data['r']) & (data['r'] < zmax)]
            dfx['phi'] = data['phi'][(zmin < data['r']) & (data['r'] < zmax)]

            dfx['r'] = data['r'][(zmin < data['r']) & (data['r'] < zmax)]
            degstheta = np.arange(0, 180, deg)
            degsphi = np.arange(-180, 180, deg)

            for jj in range(1, len(degstheta)):
                for kk in range(1, len(degsphi)):
                    dfxi = pd.DataFrame()
                    dfxi['Vgal'] = dfx['Vgal'][(degstheta[jj - 1] < dfx['theta']) & (dfx['theta'] < degstheta[jj]) & (
                                dfx['phi'] > degsphi[kk - 1]) & (dfx['phi'] < degsphi[kk])]
                    dfxi['index'] = dfx['index'][(degstheta[jj - 1] < dfx['theta']) & (dfx['theta'] < degstheta[jj]) & (
                                dfx['phi'] > degsphi[kk - 1]) & (dfx['phi'] < degsphi[kk])]
                    dfxi['x'] = dfx['x'][(degstheta[jj - 1] < dfx['theta']) & (dfx['theta'] < degstheta[jj]) & (
                                dfx['phi'] > degsphi[kk - 1]) & (dfx['phi'] < degsphi[kk])]
                    # print (dfx['x'])
                    dfxi['y'] = dfx['y'][(degstheta[jj - 1] < dfx['theta']) & (dfx['theta'] < degstheta[jj]) & (
                                dfx['phi'] > degsphi[kk - 1]) & (dfx['phi'] < degsphi[kk])]
                    dfxi['z'] = dfx['z'][(degstheta[jj - 1] < dfx['theta']) & (dfx['theta'] < degstheta[jj]) & (
                                dfx['phi'] > degsphi[kk - 1]) & (dfx['phi'] < degsphi[kk])]
                    # zmin=zmax
                    # print (dfx['theta'][(degstheta[jj-1] < dfx['theta']) & (dfx['theta'] < degstheta[jj])& (dfx['phi'] > degsphi[kk-1])& (dfx['phi'] < degsphi[kk])])

                    num.append(len(dfxi['x']))
                    # time1 = data['time'][(zmin < data['r']) & (data['r'] < zmax)]
                    # frac1= len(time1[time1 < 8.0])/len(time1)
                    # frac2= len(time1[time1 < 10.0])/len(time1)
                    # frac3= len(time1[time1 < 12.0])/len(time1)
                    # frac.append([frac1,frac2,frac3])
                    vdiffhf.append(dfxi['Vgal'].values)
                    # print (vdiffhf)
                    # spectf.append(np.concatenate(spacei))
                    indexoutf.append(dfxi['index'].values)
        numf.append(num)
        vdiffhff.append(vdiffhf)
        # print (vdiffhf)
        # spectff.append(specdto)
        indexoutff.append(indexoutf)
    return rlim, vdiffhff, numf, spectff, indexoutff, frac



def get_rstate():
    return np.random.mtrand.RandomState(seed=np.random.randint(0,2**32-1))


def mcmc_trial(data, samplenum):
    '''
    function for calculating the mcmc sample for the data
    :param data: the input data for the mcmc
    :param samplenum: sample number in mcmc
    :return: ratio betwwen the narrow and broad gaussian distribution sampled from MCMC
    '''
    mcmcf = []

    for ii in range(0, len(data)):

        res4 = optim.minimize(twogau_like, [0, 10, 0, 200, 0.5], args=(data[ii]), method='TNC',
                              bounds=[(-100, 100), (0, 100), (-100, 100), (100, 800), (1e-5, 1 - 1e-5)],
                              options={'maxfun': 1000000})

        pos = res4.x + 1e-5 * np.random.randn(32, 5)

        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=[data[ii]])

        sampler.run_mcmc(pos, samplenum, progress=False)

        # print("Took {:.1f} seconds".format(time.time()-start))

        # print(f"Now running the actual thing")

        flat_samples = sampler.get_chain(discard=2000, thin=15, flat=True)
        labels = ['center1', 'sigma1', 'center2', 'sigma2', 'ratio']
        # fig = corner.corner(
        # flat_samples, labels=labels)

        # fig.savefig('mcmc_test_halo10'+str(ii)+'.png')
        mcmc = []
        for i in range(ndim):
            mcmc.append(np.percentile(flat_samples[:, i], [32, 50, 68]))
        # fig = plt.figure(figsize =(10, 8))

        # a2,b2,c2=plt.hist(data[ii],bins=500,density = True,alpha=0.5)
        # modvals2=two_gnfunc(b2, res4.x)
        # modvals3=two_gnfunc(b2, np.array(mcmc)[:,0])
        # modvals4=two_gnfunc(b2, np.array(mcmc)[:,1])
        # plt.plot(b2,modvals2,'r',label='Least Chi Square Model')

        # plt.fill_between(b2,y1=modvals3,y2=modvals4,color='grey',zorder=2,label='MCMC Model 1-sigma Region')
        # plt.xlabel('km/s')
        # plt.ylabel('number of pairs')
        # plt.legend()
        # fig.savefig('mcmc_compare_halo10'+str(ii)+'.png')
        # plt.show()
        mcmcf.append(mcmc)
    return np.array(mcmcf)[:, 4, 1]


totmc = []
ranew = []
decnew = []
rlimf = []

for ii in np.arange(0, 360, 10):
    for jj in np.arange(-62, 80, 10):
        print(ii, jj)
        try:
            rlim, test1f = data_collect(testiron, 2 ** 8,
                                        '/Users/Jenny/thesis_idea_test/MKY_draco/rvtab-hpxcoadd-all.fits', ii, ii + 10,
                                        jj, jj + 10, 10)
            # print (len(test1f[0]),len(test1f))

            if len(test1f[0]) > 1:
                print('1')
                mci = mcmc_trial(test1f[0], 5000)
                totmc.append(mci)
                rlimf.append(rlim)
                ranew.append(ii + 5)
                decnew.append(jj + 5)

            else:
                continue
        except (ValueError):
            continue

#code for BHB sample
t = Table.read("BHB2500.txt", format="ascii.cds")
#read in the data from DESI BHB
testbhb = fits.open("Iron_BHB.fits")


def data_collect_bhb(testbhb, flag, path, ramin, ramax, decmin, decmax, scale):
    '''
    function for collecting the data from the DESI BHB sample
    :param testbhb: bhb catalog
    :param flag: target selection flag
    :param path: path for the data
    :param ramin: l min
    :param ramax: l max
    :param decmin: b min
    :param decmax: b max
    :param scale: scale around the star for collecting the data
    :return:rlim (the distance)/ test1f (vel difference between stars inside the patch and scale kpc around the star)
    '''
    ms = testbhb

    ramin = ramin * u.degree
    ramax = ramax * u.degree
    decmin = decmin * u.degree
    decmax = decmax * u.degree

    ranew = ms[2].data['TARGET_RA'][(ms[2].data['dist'] > 20)]
    decnew = ms[2].data['TARGET_DEC'][(ms[2].data['dist'] > 20)]
    dist = ms[2].data['dist'][(ms[2].data['dist'] > 20)]

    helio = SkyCoord(ra=ranew * u.degree, dec=decnew * u.degree, distance=dist * u.kpc)
    galactic2 = helio.galactic

    coord.galactocentric_frame_defaults.set('latest')
    t = pd.DataFrame()
    icrs = coord.SkyCoord(ra=ranew * u.deg, dec=decnew * u.deg,
                          radial_velocity=ms[1].data['VRAD'][(ms[2].data['dist'] > 20)] * u.km / u.s, frame='icrs')
    t['v'] = rv_to_gsr(icrs)
    gal = helio.transform_to(coord.Galactocentric)

    galc = gal.represent_as(coord.SphericalRepresentation)

    R = np.array(galc.distance.to_value(u.kpc))
    ms2 = pd.DataFrame()

    ms2['x'] = gal.cartesian.x
    ms2['y'] = gal.cartesian.y
    ms2['z'] = gal.cartesian.z
    ms2['r'] = R
    ms2['v'] = t['v']

    zmin = 20
    zmax = 100
    zlim = 4.

    t = ms2

    # cutting r and z > 20 & r < 60 kpc stars
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df1['x'] = t['x'][(zmin < t['r']) & (t['r'] < zmax)]
    df1['y'] = t['y'][(zmin < t['r']) & (t['r'] < zmax)]
    df1['z'] = t['z'][(zmin < t['r']) & (t['r'] < zmax)]
    df1['Vgal'] = t['v'][(zmin < t['r']) & (t['r'] < zmax)]
    df1['r'] = t['r'][(zmin < t['r']) & (t['r'] < zmax)]
    df2['r'] = df1['r'][(np.absolute(df1['z']) > zlim)]
    df2['x'] = df1['x'][(np.absolute(df1['z']) > zlim)]
    df2['y'] = df1['y'][(np.absolute(df1['z']) > zlim)]
    df2['z'] = df1['z'][(np.absolute(df1['z']) > zlim)]
    df2['Vgal'] = df1['Vgal'][(np.absolute(df1['z']) > zlim)]

    # data for verifying Xue 2008 et al paper
    df3 = pd.DataFrame()

    df3['x'] = t['x'][(np.absolute(t['z']) > zlim)]
    df3['y'] = t['y'][(np.absolute(t['z']) > zlim)]
    df3['z'] = t['z'][(np.absolute(t['z']) > zlim)]
    df3['Vgal'] = t['v'][(np.absolute(t['z']) > zlim)]
    df3['r'] = t['r'][(np.absolute(t['z']) > zlim)]
    df3['index'] = np.arange(0, len(df3['r']), 1)
    # print (len(df3))
    rlimf = []
    test1f = []
    numf = []
    sepf = []
    indexf = []
    fract = []
    for scale in [scale]:
        rlim, test1, num, sep, frac = v_diff(scale, df3, 10, 5000, galactic2, ramin, ramax, decmin, decmax)

        rlimf.append(rlim)
        test1f.append(test1)
        numf.append(num)
        sepf.append(sep)
        # indexf.append(indexii)
        fract.append(frac)

    return rlimf, test1f

# t
























