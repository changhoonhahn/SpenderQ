'''

methods for identifying LyA absorption 


'''
import numpy as np 
from . import util as U


def identify_LyA(wobs, fobs, iobs, zobs, wmod, fmod, sigma=2, wmax=None, method='rebin', verbose=False):
    ''' identify wavelengths with LyA absorption based on a reconstructed spectra.
    '''
    if wmax is None: 
        wmax = 1215.67 * (1. + zobs) # observed wavelength of LyA

    if method == 'rebin':
        # identify LyA regions after rebinning the observed spectra
        return _id_LyA_rebin(wobs, fobs, iobs, zobs, wmod, fmod, 
                sigma=sigma, wmax=wmax, verbose=verbose)
    else:
        raise NotImplementedError


def _id_LyA_rebin(wobs, fobs, iobs, zobs, wmod, fmod, sigma=2, wmax=None, verbose=False):
    ''' coarsely rebin spectrum and identify absorption wavelengths
    '''
    # rebin to coarser wavelength
    w_coarse = wobs[::5]
    # observed flux and ivar coarse binned
    fobs_coarse = U.trapz_rebin(wobs, fobs, edges=w_coarse)
    ivar_coarse = U.trapz_rebin(wobs, iobs/np.gradient(wobs), edges=w_coarse) * (w_coarse[1:] - w_coarse[:-1])
    # reconstructed flux coarse binned
    fmod_coarse = U.trapz_rebin(wmod, fmod, edges=w_coarse)

    is_absorb_coarse = np.zeros(len(w_coarse)-1).astype(bool)

    # below LyA
    below_lya = (w_coarse[1:] < wmax)
    is_absorb_coarse[below_lya] = (fmod_coarse[below_lya] - fobs_coarse[below_lya] > sigma * ivar_coarse[below_lya]**-0.5)
    # above LyA (more conservative 3 sigma clipping)
    is_absorb_coarse[~below_lya] = (fmod_coarse[~below_lya] - fobs_coarse[~below_lya] > 3 * ivar_coarse[~below_lya]**-0.5)

    is_absorb = np.zeros(len(wobs)).astype(bool)
    is_absorb[:-4][::5] = is_absorb_coarse
    is_absorb[1:-3][::5] = is_absorb_coarse
    is_absorb[2:-2][::5] = is_absorb_coarse
    is_absorb[3:-1][::5] = is_absorb_coarse
    is_absorb[4:][::5] = is_absorb_coarse

    return is_absorb 
