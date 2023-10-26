import numpy as np
import matplotlib.pylab as plt
import re
import sys
from astropy.io import fits
import pdb
import emcee
import corner
from scipy.optimize import curve_fit
import scipy.optimize as optim
import numpy.random as nprand

datag = np.loadtxt('/Users/Jenny/thesis_idea_test/halo/testting_g.txt')
datab = np.loadtxt('/Users/Jenny/thesis_idea_test/halo/testting_bad.txt')

xvec = np.arange(2000)-1000
mean=0
sig=50
mean2=0
sigm2=300

p0=np.asarray([mean,sig,mean2,sigm2,0.5])
gndata = np.concatenate((nprand.normal(mean,sig,size=6000),nprand.normal(mean2,sigm2,size=6000)))


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

modvals=two_gnfunc(xvec, p0)


def twogau_like(params, xvals):
    # likelihood used in fitting (normalizied)
    modelvals = two_gnfunc(xvals, params)

    mlikelihood = - np.sum(np.log(modelvals))

    # print(mlikelihood)
    return (mlikelihood)

res4 = optim.minimize(twogau_like,[0,10,0,200,0.5],args=(datab),method = 'TNC',
                    bounds=[(-100,100),(0,300),(-100,100),(50,800),(1e-5,1-1e-5)],
                    options={'maxfun':1000000})

a2,b2,c2=plt.hist(datab,bins=500,density = True)
modvals2=two_gnfunc(b2, res4.x)
plt.plot(b2,modvals2,'r')
plt.show()

def log_prior(theta):
    m,b,log_f,c,k = theta
    if  -50 < m < 50 and 0 < b < 120 and -50 < log_f < 50 and 100 < c < 500 and 1e-5 < k < 1-1e-5:
        return 0.0
    return -np.inf

def log_probability(theta, x):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return -lp -  twogau_like(theta, x)


def mcmc_trial(data, samplenum):
    for ii in range(0, len(data)):

        res4 = optim.minimize(twogau_like, [0, 10, 0, 200, 0.5], args=(data[ii]), method='TNC',
                              bounds=[(-100, 100), (0, 100), (-100, 100), (100, 800), (1e-5, 1 - 1e-5)],
                              options={'maxfun': 1000000})

        pos = res4.x + 1e-5 * np.random.randn(32, 5)
        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=[data[ii]])

        sampler.run_mcmc(pos, samplenum, progress=True)

        flat_samples = sampler.get_chain(discard=2000, thin=15, flat=True)
        labels = ['center1', 'sigma1', 'center2', 'sigma2', 'ratio']
        fig = corner.corner(
            flat_samples, labels=labels)

        fig.savefig('mcmc_test_halo10' + str(ii) + '.png')
        mcmc = []
        for i in range(ndim):
            mcmc.append(np.percentile(flat_samples[:, i], [32, 68]))
        fig = plt.figure(figsize=(10, 8))

        a2, b2, c2 = plt.hist(data[ii], bins=500, density=True)
        modvals2 = two_gnfunc(b2, res4.x)
        modvals3 = two_gnfunc(b2, np.array(mcmc)[:, 0])
        modvals4 = two_gnfunc(b2, np.array(mcmc)[:, 1])
        plt.plot(b2, modvals2, 'r', label='Least Chi Square Model')

        plt.fill_between(b2, y1=modvals3, y2=modvals4, color='k', zorder=2, label='MCMC Model 1-sigma Region')
        plt.xlabel('km/s')
        plt.ylabel('number of pairs')
        plt.legend()
        fig.savefig('mcmc_compare_halo10' + str(ii) + '.png')
        plt.show()

fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = [ 'center1','sigma1','center2','sigma2','ratio']
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(500, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
res4 = optim.minimize(twogau_like,[0,10,0,200,0.5],args=(datab),method = 'TNC',
                    bounds=[(-100,100),(0,100),(-100,100),(100,800),(1e-5,1-1e-5)],
                    options={'maxfun':1000000})
print (res4)
pos = res4.x + 1e-5 * np.random.randn(32, 5)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=[datab])

sampler.run_mcmc(pos, 20000, progress=True)


a2,b2,c2=plt.hist(datab,bins=500,density = True)
flat_samples = sampler.get_chain(discard=2000, thin=15, flat=True)
print(flat_samples.shape)


mcmc=[]
for i in range(ndim):
    mcmc.append(np.percentile(flat_samples[:, i], [32,68]))

modvals2=two_gnfunc(b2, res4.x)
modvals3=two_gnfunc(b2, np.array(mcmc)[:,0])
modvals4=two_gnfunc(b2, np.array(mcmc)[:,1])
plt.plot(b2,modvals2,'r')

plt.fill_between(b2,y1=modvals3,y2=modvals4,color='k',zorder=2)
plt.show()


















