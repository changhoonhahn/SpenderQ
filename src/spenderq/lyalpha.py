'''

methods for identifying LyA absorption 


'''
import numpy as np 
from . import util as U


def identify_LyA(wobs, fobs, iobs, zobs, wmod, fmod, sigma=1.5, wmax=None, method='rebin', verbose=False):
    ''' identify wavelengths with LyA absorption based on a reconstructed spectra.
    '''
    if wmax is None: 
        wmax = 1215.67 * (1. + zobs) # observed wavelength of LyA


    if method == 'rebin':
        # identify LyA regions after rebinning the observed spectra to 4A
        w_coarse, fobs_coarse, iobs_coarse, fmod_coarse = rebin(wobs, fobs, iobs, zobs, wmod, fmod, 
                method='fixed', wmax=wmax, verbose=verbose)
    elif method == 'snr_rebin': 
        # identify LyA regions after rebinning the observed spectra based on SNR
        w_coarse, fobs_coarse, iobs_coarse, fmod_coarse = rebin(wobs, fobs, iobs, zobs, wmod, fmod, 
                method='uniform', wmax=wmax, verbose=verbose)
    else:
        raise NotImplementedError

    is_absorb_coarse = np.zeros(len(fobs_coarse)).astype(bool)
    # below LyA
    below_lya = (w_coarse[1:] < wmax) # right edge of spectral element is less wmax
    is_absorb_coarse[below_lya] = (fmod_coarse[below_lya] - fobs_coarse[below_lya] > sigma * iobs_coarse[below_lya]**-0.5)
    # above LyA (more conservative 3 sigma clipping)
    is_absorb_coarse[~below_lya] = (fmod_coarse[~below_lya] - fobs_coarse[~below_lya] > 3 * iobs_coarse[~below_lya]**-0.5)

    i_coarse = np.digitize(wobs, w_coarse, right=False) - 1
    
    is_absorb = np.zeros(len(wobs)).astype(bool)  
    is_absorb = is_absorb_coarse[i_coarse]
    return is_absorb


def rebin(wobs, fobs, iobs, zobs, wmod, fmod, method='uniform', wmax=None, verbose=True):
    if method == 'fixed': 
        w_coarse = wrebin_fixed(wobs, fobs, iobs, zobs, wmax=wmax)
    elif method == 'uniform': 
        w_coarse = wrebin_uniform(wobs, fobs, iobs, zobs, wmax=wmax)
    else: 
        raise NotImplementedError 
    if verbose: print('rebinning to Nbin = %i' % len(w_coarse))

    # observed flux and ivar coarse binned
    fobs_coarse = np.zeros(len(w_coarse)-1)
    iobs_coarse = np.zeros(len(w_coarse)-1)
    fobs_coarse[1:-1] = U.trapz_rebin(wobs, fobs, edges=w_coarse[1:-1])
    iobs_coarse[1:-1] = U.trapz_rebin(wobs, iobs/np.gradient(wobs), 
                                         edges=w_coarse[1:-1])

    # deal with edges (assume uniform binning for observed anad model spectra)
    wlim = (wobs < w_coarse[1])
    fobs_coarse[0] = np.sum(np.diff(wobs)[0] * fobs[wlim])/(w_coarse[1] - w_coarse[0])
    iobs_coarse[0] = np.sum(np.diff(wobs)[0] * (iobs/np.gradient(wobs))[wlim])/(w_coarse[1] - w_coarse[0])
    wlim = (wobs > w_coarse[-2])
    fobs_coarse[-1] = np.sum(np.diff(wobs)[0] * fobs[wlim])/(w_coarse[-1] - w_coarse[-2])
    iobs_coarse[-1] = np.sum(np.diff(wobs)[0] * (iobs/np.gradient(wobs))[wlim])/(w_coarse[-1] - w_coarse[-2])
    # rescale ivar
    iobs_coarse *= np.diff(w_coarse)
    
    # reconstructed flux coarse binned
    fmod_coarse = np.zeros(len(w_coarse)-1)
    fmod_coarse[1:-1] = U.trapz_rebin(wmod, fmod, edges=w_coarse[1:-1])
    
    wlim = (w_coarse[0] <= wmod) & (wmod < w_coarse[1])
    fmod_coarse[-1] = np.sum(np.diff(wmod)[0] * fmod[wlim])/(w_coarse[1] - w_coarse[0])
    wlim = (w_coarse[-2] <= wmod) & (wmod < w_coarse[-1])
    fmod_coarse[-1] = np.sum(np.diff(wmod)[0] * fmod[wlim])/(w_coarse[-1] - w_coarse[-2])
    return w_coarse, fobs_coarse, iobs_coarse, fmod_coarse


def wrebin_fixed(wobs, fobs, iobs, zobs, wmax=None):
    ''' uniform rebinning where binsize is fixed to 4A 
    '''
    dw = 4.0

    # rebinning 
    Nbin = int(((wobs[-1] - wobs[0]) + 0.8)/dw)
    w_coarse = np.linspace(wobs[0]-0.4, wobs[-1]+0.4, Nbin)
    return w_coarse


def wrebin_uniform(wobs, fobs, iobs, zobs, wmax=None):
    ''' uniform rebinning where binsize is scaled by SNR over lambda < wmax  
    '''
    # number of bins based on overall snr
    snr = fobs * iobs**0.5

    # scale resolution by SNR of the spectra below LyA
    # the scaling is set so that at SNR = 1, each spectral element is 8A and the minimum 
    # bin width is 4A. 
    dw = np.clip(8.0 / np.median(snr[wobs < wmax]), 4.0, 16.)

    # coarse binning
    Nbin = int(((wobs[-1] - wobs[0]) + 0.8)/dw)
    w_coarse = np.linspace(wobs[0]-0.4, wobs[-1]+0.4, Nbin)
    return w_coarse
