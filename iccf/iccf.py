# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the cross-correlation function from White & Peterson (1994)
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import argparse, sys

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
        t, y, yerr = np.genfromtxt(fname, usecols = (0, 1, 2), delimiter = ',', dtype = np.float64, unpack = True)
    except Exception, e:
        print('Could not read file '+str(fname))
        sys.exit()
    return {'t': t, 'y': y, 'yerr': yerr}

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

def plot_original(x, y, c, title = None):
    """ Show original light curves
        
    Parameters
    ----------
    
        x     (array): Epoch
        y     (array): Flux measurement
        c     (array): Associated flux uncertainty
        title (str)  : Continuum or line-emission   
        
    Returns
    -------
    
        Plotted light curve with error bars
    """
    plt.figure()

    plt.errorbar(x, y, yerr = c, fmt = 'o', color = 'black')
    plt.title(title, fontsize = 'x-large')
    plt.xlabel('HJD - 2450000', fontsize = 'x-large')
    plt.ylabel('Flux', fontsize = 'x-large')
    plt.grid()

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
    mask = (corr > corr.max() * threshold/100.0)
    above_corr = corr[mask]
    above_lags = lags[mask]
    
    # Compute the centroid of the peak
    centroid = (above_lags * above_corr).sum() / above_corr.sum()

    # The lag of the cross-correlation peak
    peak = lags[np.argmax(corr)]
    
    return {'centroid': centroid, 'peak': peak}

def show_CCF(lags, corr, title = None):
    """ Show cross-correlation results
        
    Parameters
    ----------
    
        lags  (array): Lag values
        corr  (array): cross-correlation values
        
    Returns
    -------
    
        Plotted cross-correlation results
    """
    plt.figure()

    plt.plot(lags, corr, '-k')
    plt.ylim([0, 1])
    plt.title(title)
    plt.xlabel('lags [days]', fontsize = 'x-large')
    plt.ylabel('r', fontsize = 'x-large')
    plt.grid()

def calc_uncertaintities(results, title = None):
    """ Calculate upper and lower uncertainties on the centroid or peak
        
    Parameters
    ----------
    
        results (array): Centroid or peak distribution
        title   (str)  : Centroid or Peak
        
    Returns
    -------
    
        The median and associated upper and lower uncertainties
    """
    # Sort the array
    sorted = np.sort(results)
    
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

    return {'t': new_t, 'y': new_y}

def show_CCPD(lags, results, title = None):
    """ Plot the centroid or peak distribution
        
    Parameters
    ----------
    
        results (array): Centroid or peak distribution
        lags    (array): Lag values
        title   (str)  : Appropriate plot title

    Returns
    -------
    
        cross-correlation centroid distribution (CCCD) or cross-correlation peak distribution (CCPD)
    """
    plt.figure()
    
    plt.hist(results, lags, normed = True, histtype = 'step', color = 'black')
    plt.xlabel('Lags [days]', fontsize = 'x-large')
    plt.ylabel('Probability', fontsize = 'x-large')
    plt.title(title, fontsize = 'x-large')

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

    return {'y': y_rm_dupl_noise, 't': t_rm_dupl}

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
        line = subset(tx, x, xerr)
        cont = subset(ty, y, yerr)
        
        # Cross-correlation
        first_pass = cross_corr(line['y'], line['t'], cont['y'], cont['t'], minlag, maxlag, step, shifted = 'C')
        second_pass = cross_corr(line['y'], line['t'], cont['y'], cont['t'], minlag, maxlag, step, shifted = 'L')

        avg_corr = np.mean(np.array([first_pass, second_pass]), axis = 0)
        lags = np.arange(minlag, maxlag + step, step)

        
        # Get the lags
        results = get_lags(lags, avg_corr, threshold = 80)
        
        # Append
        centroids[index] = results['centroid']
        peaks[index] = results['peak']

    return {'centroids': centroids, 'peaks': peaks}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'This program employs the Interpolation Cross-Correlation method (ICCF) developed by Gaskell & Sparke (1986), as implemented by White & Peterson (1994).')

    parser.add_argument('-c', '--cont', action = 'store', help = 'The continuum light curve C(t)', type = str)
    parser.add_argument('-l', '--line', action = 'store', help = 'The emission-line light curve L(t)', type = str)
    parser.add_argument('lgl', action = 'store', help = 'Lag range low', type = float)
    parser.add_argument('lgh', action = 'store', help = 'Lag range high', type = float)
    parser.add_argument('dt', action = 'store', help = 'Lag bin width', type = float)

    args = parser.parse_args()

    # Read user files

    line = read(args.line)
    cont = read(args.cont)

    cont_t, cont_f = cont['t'], mag_to_flux(cont['y'])
    cont_err = to_flux_err(cont['yerr'], cont_f)
    line_t, line_f, line_err = line['t'], line['y'], line['yerr']

    # Plot original light curves

    plot_original(cont_t, cont_f, cont_err, title = 'Continuum')
    plot_original(line_t, line_f, line_err, title = 'Line')
    
    # Interpolation

    interp_cont = interpolation(cont_t, cont_f, args.dt)
    interp_line = interpolation(line_t, line_f, args.dt)

    # Lag range
    lags = np.arange(args.lgl, args.lgh + args.dt, args.dt)
    
    # Auto-correlation

    acf_cont = cross_corr(interp_cont['y'], interp_cont['t'], interp_cont['y'], interp_cont['t'], args.lgl, args.lgh, args.dt, shifted = 'C')
    acf_line = cross_corr(interp_line['y'], interp_line['t'], interp_line['y'], interp_line['t'], args.lgl, args.lgh, args.dt, shifted = 'L')

    show_CCF(lags, acf_cont, title = 'Continuum ACF')
    show_CCF(lags, acf_line, title = 'Line ACF')

    # Cross-correlation

    first_pass = cross_corr(line_f, line_t, cont_f, cont_t, args.lgl, args.lgh, args.dt, shifted = 'C')
    second_pass = cross_corr(line_f, line_t, cont_f, cont_t, args.lgl, args.lgh, args.dt, shifted = 'L')
    
    avg_corr = np.mean(np.array([first_pass, second_pass]), axis = 0)
    
    show_CCF(lags, avg_corr, title = 'CCF')

    orig_lags = get_lags(lags, avg_corr, threshold = 80)

    print('The centroid of the original data: {} days'.format(orig_lags['centroid']))
    print('The peak of the original data: {} days'.format(orig_lags['peak']))

    results = FR_RSS(line_f, line_t, line_err, cont_f, cont_t, cont_err, args.lgl, args.lgh, args.dt, num = 1000)
    
    calc_uncertaintities(results['centroids'], title = 'Centroid')
    calc_uncertaintities(results['peaks'], title = 'Peaks')
    
    show_CCPD(lags, results['centroids'], title = r'$\tau_{cent}$')
    show_CCPD(lags, results['peaks'], title = r'$\tau_{peak}$')
    
    plt.show()
