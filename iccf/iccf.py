# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the cross-correlation function from White & Peterson (1994)
from __future__ import (absolute_import, division, print_function, unicode_literals)

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats
import argparse, sys

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12

def read(fname):
    """ Reads both continuum and line-emission files passed by the user.
        
    Parameters
    ----------
    
        fname (str): File name, columns must be comma separated
    
    Raises
    ------
    
        Raises error if file is not found and exits the code
        
    Returns:
    -------
    
        t    (array): Observation date
        y    (array): Magnitude or flux
        yerr (array): Associated uncertainties
    """
    try:
        t, y, ye = np.genfromtxt(fname, usecols = (0, 1, 2), dtype = np.float64, unpack = True)
    except Exception, e:
        print('Could not read file '+str(fname))
        sys.exit()
    return t, y, ye

def mag_to_flux(mag):
    """ Convert V-band magnitude to flux.
        
    Parameters
    ----------
    
        mag  (array): Magnitude
        
    Returns
    -------
    
        flux (array): In units of erg/s/cm2/AA
    
    Reference
    ---------
    
        - Campins, Rieke & Lebofsky 1985, AJ, 90, 896
        - Rieke, Lebofsky & Low 1985, AJ, 90, 900
        - Allen's Astrophysical Quantities, Fourth Edition, 2001, Arthur N.Cox(ed.), Springer-Verlag
    """
    return (3e-13 * 3781 * 10 ** (-0.4 * mag)) / (0.55 ** 2)

def to_flux_err(err_mag, counts):
    """ Convert magnitude errors to flux errors
        
    Parameters
    ----------
    
        err_mag     (array): Magnitude errors to be converted
        counts      (array): Flux
        
    Returns
    -------
    
        flux errors (array): In units of erg/s/cm2/AA
        
    Reference
    ---------
    
        See notes on photometric flux calibration:
        http://classic.sdss.org/dr7/algorithms/fluxcal.html
    """
    return (err_mag * np.log(10) * counts) / 2.5

def plot_original(x, y, ye, x2, y2, ye2, t = None, t2 = None):
    """ Show original light curves
        
    Parameters
    ----------
    
        x      (array): Epoch
        y      (array): Flux measurements
        ye     (array): Associated flux uncertainty
        t      (str)  : Title of the plot
        
    Returns
    -------
    
        Plotted light curve with error bars
    """
    fig = plt.figure()

    axis1 = fig.add_subplot(211)
    axis1.errorbar(x, y, yerr = ye, fmt = 'o', color = 'black', label = t)
    axis1.legend(loc = 'best')
    axis2 = fig.add_subplot(212)
    axis2.errorbar(x2, y2, yerr = ye2, fmt = 'o', color = 'green', label = t2)
    axis2.set_xlabel('HJD - 2450000')
    axis2.legend(loc = 'best')
    fig.text(0.03, 0.5, 'Flux', ha='center', va='center', rotation='vertical', fontweight = 'bold')

def cross_corr(x, tx, y, ty, minlag, maxlag, step, shifted = None):
    """ Calculate the cross-correlation series between two light curves
        
    Parameters
    ----------
    
        x       (array): Flux of the emission-line
        tx      (array): Corresponding observation dates
        y       (array): Flux of the continuum-emission
        ty      (array): Corresponding observation dates
        minlag  (float): Lag range low (days)
        maxlag  (float): Lag range high (days)
        step    (float): Lag bin width (days)
        shifted   (str): Light curve to be shifted, C for continuum and L for line
        
    Returns
    -------
    
        lags    (array): Lag values in days
        corr    (array): Cross-correlation values

    Reference
    ---------
    
        See online documentation based on White & Peterson (1994)
    """
    # Initialize the output arrays
    lags = np.arange(minlag, maxlag + step, step)
    corr = np.zeros(lags.size, dtype = np.float64)

    # Calculate the correlation series
    for index, lag in enumerate(lags):
        if shifted == 'C':  # Shift and interpolate the continuum
            mask = (tx-lag > tx[0]) * (tx-lag < tx[-1])
            ynew = np.interp(tx[mask] - lag, ty, y)
            xnew = 1.0 * x[mask]
        if shifted == 'L':  # Shift and interpolate the line
            mask = (ty+lag > ty[0]) * (ty+lag < ty[-1])
            xnew = np.interp(ty[mask] + lag, tx, x)
            ynew = 1.0 * y[mask]

        # Unbias the signal by subtracting the mean
        xdiff = xnew - xnew.mean()
        ydiff = ynew - ynew.mean()
        
        # Compute the normalized CCF
        corr[index] = scipy.stats.pearsonr(xdiff, ydiff)[0]

    # Return correlation coefficients
    return corr

def get_lags(lags, corr, threshold = 80):
    """ Calculate the centroid and the peak
        
    Parameters
    ----------
    
        lags      (array): Lag values
        corr      (array): Cross-correlation values
        threshold (int)  : Only select values above the specified threshold (80 or 90). For much broader peaks, 90 is usually preferred
        
    Returns
    -------
    
        centroid  (float): Centroid of the peak
        peak      (float): Peak
    """
    # Get correlations above specified threshold and corresponding lags
    mask = (corr >= corr.max() * threshold/100.0)
    above_corr = corr[mask]
    above_lags = lags[mask]
    
    if above_corr.size > 0:
        # Compute the centroid of the peak
        centroid = (above_lags * above_corr).sum() / above_corr.sum()
    else:
        centroid = 0

    # The lag of the cross-correlation peak
    peak = lags[np.argmax(corr)]
    
    return centroid, peak

def show_CCF(avg_lags, avg_corr, acf_lags, acf_corr, centroid, peak):
    """ Show cross-correlation results
        
    Parameters
    ----------
    
        avg_lags  (array): Average lag values
        avg_corr  (array): Average cross-correlation values
        acf_lags  (array): Autocorrelated lag values
        acf_corr  (array): Autocorrelated cross-correlation values
        centroid  (float): Centroid of the peak
        peak      (float): Peak of the CCF
        
    Returns
    -------
    
        Plotted cross-correlation results
    """
    plt.figure()

    plt.plot(acf_lags, acf_corr, linestyle = '-', marker = '.', color = 'black', label = 'ACF')
    plt.xlabel('lags [days]')
    plt.ylabel('r')
    plt.legend(loc = 'best')
    
    plt.figure()
    
    plt.plot(avg_lags, avg_corr, linestyle = '-', marker = '.', color = 'black', label = 'Average')
    
    mask = (avg_corr >= 0.8 * avg_corr.max())
    above_corr = avg_corr[mask]
    above_lags = avg_lags[mask]
    
    if above_corr.size > 0:
        for idx in range(len(above_corr)):
            plt.plot([above_lags[idx], above_lags[idx]], [above_corr[idx], 0.8 * above_corr.max()], color = 'red')
    
        plt.plot([centroid, centroid], [above_corr.min(), avg_corr.min()], linestyle = '--', color = 'red')
        plt.annotate(r'%.1f' % (centroid), xy = (centroid + 2 * args.dt, avg_corr.min() + 0.2))
        plt.ylim([avg_corr.min(), avg_corr.max()])
        
    plt.xlabel('lags [days]')
    plt.ylabel('r')
    plt.legend(loc = 'best')

def calc_uncertaintities(ccf, title = None):
    """ Calculate upper and lower uncertainties on the centroid or peak
        
    Parameters
    ----------
    
        ccf   (array): Centroid or peak distribution
        title (str)  : Centroid or Peak
        
    Returns
    -------
    
        The median and associated upper and lower uncertainties
    """
    # Sort the array
    sorted = np.sort(ccf)
    
    # Get the median of the distribution
    lag = np.percentile(sorted, 50)
    
    # Get corresponding 1 sigma errors
    plus_sigma = np.percentile(sorted, 84)
    minus_sigma = np.percentile(sorted, 16)
    
    # Upper and lower uncertainties
    upper = plus_sigma - lag
    lower = minus_sigma - lag

    print('--------------------{}-----------------------'.format(title))
    print('The upper error: {} days'.format(upper))
    print('The lower error: {} days'.format(lower))
    print('The median: {} days'.format(lag))

    return lag, upper, lower

def interpolation(t, y, dy):
    """ Put on a uniform time grid by interpolating
        
    Parameters
    ----------
    
        t  (array): Epoch
        y  (array): Flux
        dy (float): time step
        
    Returns
    -------
    
        One-dimensional linear interpolation
    """
    new_t = np.arange(t[0], t[-1] + dy, dy)
    new_y = np.interp(new_t, t, y)

    return new_t, new_y

def show_CCPD(lags, ccf, lag, up, lw, title = None):
    """ Plot the centroid or peak distribution
        
    Parameters
    ----------
    
        ccf     (array): Centroid or peak distribution
        lags    (array): Lag values
        lag     (float): Median of the distribution
        up      (float): Upper error
        lw      (float : Lower error
        title   (str)  : Appropriate plot title

    Returns
    -------
    
        cross-correlation centroid distribution (CCCD) or cross-correlation peak distribution (CCPD)
    """
    plt.figure()
    
    n, bins, patches = plt.hist(ccf, lags, normed = True, histtype = 'step', color = 'black')
    
    mask = ((lags > lag + lw) & (lags < lag)) | ((lags > lag) & (lags < (lag + up)))
    lags_masked = lags[mask]
    indices = np.searchsorted(lags, lags_masked)

    if indices.size > 0:
        for idx in indices:
            plt.plot([lags[idx], lags[idx]], [0, n[idx]], linestyle = '--', color = 'red')
    
        plt.plot([lag, lag], [0, np.interp(lag, lags[indices], n[indices])], linewidth = 3, color = 'red')
    plt.xlabel('Lags [days]')
    plt.ylabel('Probability')
    plt.title(title)

# Error estimation using flux randomization and random subset selection method

def subset(t, y, yerr):
    """ y is modified by adding gaussian noise based on the quoted uncertainty on y
        
    Parameters
    ----------
        
        t    (array): Observation date
        y    (array): flux
        yerr (array): Associated uncertainties
        
    Returns
    -------
    
        modified y
        
    Reference
    ---------
    
        Peterson (1998)
    """
    # Random subset selection with replacement
    t_rand_samp = np.random.choice(t, t.size, replace = True)

    # Remove duplicates
    t_rm_dupl = np.unique(t_rand_samp)

    # Sort
    t_rm_dupl.sort()

    # Get indices
    indices = np.searchsorted(t, t_rm_dupl)

    # Get fluxes
    y_rm_dupl = y[indices]

    # Get corresponding error fluxes
    yerr_rm_dupl = yerr[indices]

    # Add Gaussian noise
    y_rm_dupl_noise = np.random.normal(y_rm_dupl, yerr_rm_dupl)

    return y_rm_dupl_noise, t_rm_dupl

def FR_RSS(x, tx, xerr, y, ty, yerr, minlag, maxlag, step, num = None):
    """ Calculate the centroid and peak distributions
        
    Parameters
    ----------
    
        x       (array): Flux of the emission-line
        tx      (array): Corresponding observation dates
        xerr    (array): Associated uncertainties
        y       (array): Flux of the continuum-emission
        ty      (array): Corresponding observation dates
        yerr    (array): Associated uncertainties
        minlag  (float): Lag range low (days)
        maxlag  (float): Lag range high (days)
        step    (float): Lag bin width (days)
        num     (int)  : Number of simulations/realizations
        
    Returns
    -------
    
        Centroid and peak distributions
    """
    # Array to store all centroids
    centroids = np.zeros(num, dtype = np.float64)
    
    # Array to store all peaks
    peaks = np.zeros(num, dtype = np.float64)

    for index in range(num):
        # Random subset selection
        sub_ln_y, sub_ln_t = subset(tx, x, xerr)
        sub_ct_y, sub_ct_t = subset(ty, y, yerr)
        
        # Cross-correlation
        first_pass = cross_corr(sub_ln_y, sub_ln_t, sub_ct_y, sub_ct_t, minlag, maxlag, step, shifted = 'C')
        second_pass = cross_corr(sub_ln_y, sub_ln_t, sub_ct_y, sub_ct_t, minlag, maxlag, step, shifted = 'L')

        avg_corr = np.mean(np.array([first_pass, second_pass]), axis = 0)
        lags = np.arange(minlag, maxlag + step, step)

        
        # Get the lags
        cen, pk = get_lags(lags, avg_corr, threshold = 80)
        
        # Append
        centroids[index] = cen
        peaks[index] = pk

    return centroids, peaks


if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'This program employs the Interpolation Cross-Correlation method (ICCF) developed by Gaskell & Sparke (1986), as implemented by White & Peterson (1994).')

    parser.add_argument('-c', '--cont', action = 'store', help = 'The continuum light curve C(t)', type = str)
    parser.add_argument('-l', '--line', action = 'store', help = 'The emission-line light curve L(t)', type = str)
    parser.add_argument('lgl', action = 'store', help = 'Lag range low', type = float)
    parser.add_argument('lgh', action = 'store', help = 'Lag range high', type = float)
    parser.add_argument('dt', action = 'store', help = 'Lag bin width', type = float)

    args = parser.parse_args()

    # Read user files
    ln_t, ln_y, ln_ye = read(args.line)
    ct_t, ct_y, ct_ye = read(args.cont)
    
    prompt_user = raw_input('Is the continuum expressed in magnitude (y/n): ')

    if prompt_user == 'y':
        ct_y = mag_to_flux(ct_y)
        ct_ye = to_flux_err(ct_ye, ct_y)

    # Plot original light curves
    plot_original(ct_t, ct_y, ct_ye, ln_t, ln_y, ln_ye, t = 'Continuum', t2 = 'Line')
    
    # Interpolation
    interp_ct_t, interp_ct_y = interpolation(ct_t, ct_y, args.dt)

    # Lag range
    lags = np.arange(args.lgl, args.lgh + args.dt, args.dt)

    # Auto-correlation
    acf_cont = cross_corr(interp_ct_y, interp_ct_t, interp_ct_y, interp_ct_t, args.lgl, args.lgh, args.dt, shifted = 'C')

    # Cross-correlation
    first_pass = cross_corr(ln_y, ln_t, ct_y, ct_t, args.lgl, args.lgh, args.dt, shifted = 'C')
    second_pass = cross_corr(ln_y, ln_t, ct_y, ct_t, args.lgl, args.lgh, args.dt, shifted = 'L')
    
    avg_corr = np.mean(np.array([first_pass, second_pass]), axis = 0)

    # Get centroid and peak
    cen, pk = get_lags(lags, avg_corr, threshold = 80)

    # Plot
    show_CCF(lags, avg_corr, lags, acf_cont, cen, pk)

    print('The centroid of the original data: '+str(cen))
    print('The peak of the original data: '+str(pk))
    
    # Get errors
    centroids, peaks = FR_RSS(ln_y, ln_t, ln_ye, ct_y, ct_t, ct_ye, args.lgl, args.lgh, args.dt, num = 1000)
    
    cen_med, cen_up, cen_lw = calc_uncertaintities(centroids, title = 'Centroid')
    pk_med, pk_up, pk_lw = calc_uncertaintities(peaks, title = 'Peaks')

    # Show distributions
    show_CCPD(lags, centroids, cen_med, cen_up, cen_lw, title = r'$\tau_{cent}$')
    show_CCPD(lags, peaks, pk_med, pk_up, pk_lw, title = r'$\tau_{peak}$')
    
    # Save to PDF file
    pdf = PdfPages('test.pdf')
    
    for fig in xrange(1, plt.figure().number): # will open an empty extra figure :(
        pdf.savefig( fig )
    pdf.close()
