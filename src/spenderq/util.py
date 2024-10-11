import os
import numba
import numpy as np
import scipy.sparse


class london_picca(object): 
    def __init__(self, line): 
        ''' simple object to interface with London Picca continuum fits 
        '''
        import fitsio

        # read picca continuum
        if line == 'LyA': 
            fdelta = '/tigress/chhahn/spender_qso/picca/delta_attributes.fits.gz'
        elif line == 'LyB': 
            fdelta = '/tigress/chhahn/spender_qso/picca/delta_attributes_iteration4_londonlyb.fits.gz'
        else: 
            raise ValueError('line has to be either LyA or LyB') 


        with fitsio.FITS(fdelta) as attrs:

            cont = attrs['CONT'].read()
            loglam = cont['LOGLAM_REST']
            lamb_rest=10**loglam
            mean_cont   = cont['MEAN_CONT']
            cut = cont['WEIGHT']>0.

            self.wave_picca = lamb_rest[cut]

            self.mean_continuum_picca = mean_cont[cut]

            if 'FIT_METADATA' not in attrs: raise ValueError 

            self.picca_cont = attrs['FIT_METADATA'].read()

    def get_continuum(self, tid):
        ''' get picca continuum given target id
        '''
        # plot picca continuum given london mock target id
        w = (self.picca_cont['LOS_ID'] == tid)

        if sum(w) > 0 and self.picca_cont['ACCEPTED_FIT'][w]==True:
            aq = self.picca_cont['ZERO_POINT'][w]
            bq = self.picca_cont['SLOPE'][w]

            if aq < 0: print('negative slope')

            lambda_func = np.log10(self.wave_picca/min(self.wave_picca))
            lambda_func /= np.log10(max(self.wave_picca)/min(self.wave_picca))
            fitted_continuum = self.mean_continuum_picca * (lambda_func * bq + aq)
        else:
            print('no fit')
            return None

        return fitted_continuum


def centers2edges(centers):
    """Convert bin centers to bin edges, guessing at what you probably meant
    Args:
        centers (array): bin centers,
    Returns:
        array: bin edges, lenth = len(centers) + 1
    """
    centers = np.asarray(centers)
    edges = np.zeros(len(centers)+1)
    #- Interior edges are just points half way between bin centers
    edges[1:-1] = (centers[0:-1] + centers[1:]) / 2.0
    #- edge edges are extrapolation of interior bin sizes
    edges[0] = centers[0] - (centers[1]-edges[1])
    edges[-1] = centers[-1] + (centers[-1]-edges[-2])

    return edges

@numba.jit
def _trapz_rebin(x, y, edges, results):
    '''
    Numba-friendly version of trapezoidal rebinning
    See redrock.rebin.trapz_rebin() for input descriptions.
    `results` is pre-allocated array of length len(edges)-1 to keep results
    '''
    nbin = len(edges) - 1
    i = 0  #- index counter for output
    j = 0  #- index counter for inputs
    yedge = 0.0
    area = 0.0

    while i < nbin:
        #- Seek next sample beyond bin edge
        while x[j] <= edges[i]:
            j += 1

        #- What is the y value where the interpolation crossed the edge?
        yedge = y[j-1] + (edges[i]-x[j-1]) * (y[j]-y[j-1]) / (x[j]-x[j-1])

        #- Is this sample inside this bin?
        if x[j] < edges[i+1]:
            area = 0.5 * (y[j] + yedge) * (x[j] - edges[i])
            results[i] += area

            #- Continue with interior bins
            while x[j+1] < edges[i+1]:
                j += 1
                area = 0.5 * (y[j] + y[j-1]) * (x[j] - x[j-1])
                results[i] += area

            #- Next sample will be outside this bin; handle upper edge
            yedge = y[j] + (edges[i+1]-x[j]) * (y[j+1]-y[j]) / (x[j+1]-x[j])
            area = 0.5 * (yedge + y[j]) * (edges[i+1] - x[j])
            results[i] += area

        #- Otherwise the samples span over this bin
        else:
            ylo = y[j] + (edges[i]-x[j]) * (y[j] - y[j-1]) / (x[j] - x[j-1])
            yhi = y[j] + (edges[i+1]-x[j]) * (y[j] - y[j-1]) / (x[j] - x[j-1])
            area = 0.5 * (ylo+yhi) * (edges[i+1]-edges[i])
            results[i] += area

        i += 1

    for i in range(nbin):
        results[i] /= edges[i+1] - edges[i]

    return

def trapz_rebin(x, y, xnew=None, edges=None):
    """Rebin y(x) flux density using trapezoidal integration between bin edges
    Notes:
        y is interpreted as a density, as is the output, e.g.
        >>> x = np.arange(10)
        >>> y = np.ones(10)
        >>> trapz_rebin(x, y, edges=[0,2,4,6,8])  #- density still 1, not 2
        array([ 1.,  1.,  1.,  1.])
    Args:
        x (array): input x values.
        y (array): input y values.
        edges (array): (optional) new bin edges.
    Returns:
        array: integrated results with len(results) = len(edges)-1
    Raises:
        ValueError: if edges are outside the range of x or if len(x) != len(y)
    """
    if edges is None:
        edges = centers2edges(xnew)
    else:
        edges = np.asarray(edges)

    if edges[0] < x[0] or x[-1] < edges[-1]:
        raise ValueError('edges must be within input x range')

    result = np.zeros(len(edges)-1, dtype=np.float64)

    _trapz_rebin(x, y, edges, result)

    return result
