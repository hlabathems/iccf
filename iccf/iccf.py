#!/usr/bin/env python
# Author: Michael Hlabathe
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import argparse, sys

def read(fname):
    """
    Args:
        fname (str): File name, columns must be comma separated
    Returns:
        t    (array): Observation date
        y    (array): Magnitude or flux
        yerr (array): Associated uncertainties or errors
    """
    try:
        t, y, yerr = np.genfromtxt(fname, usecols = (0, 1, 2), delimiter = ',', dtype = np.float64, unpack = True)
    except Exception, e:
        print 'Could not read file '+str(fname)
        sys.exit()
    return {'t': t, 'y': y, 'yerr': yerr}

def mag_to_flux(mag):
    """ Convert V-band magnitude to flux
    Args:
        mag  (array): Magnitude
    Returns:
        flux (array): In units of erg/s/cm2/AA
    
    References      : http://ssc.spitzer.caltech.edu/warmmission/propkit/pet/magtojy/ref.html AND http://www.stsci.edu/hst/nicmos/documents/handbooks/current_NEW/Appendix_B.14.3.html
    """
    return (3e-13 * 3781 * 10 ** (-0.4 * mag)) / (0.55 ** 2)

def to_flux_err(err_mag, counts):
    """ Convert magnitude errors to flux errors
    Args:
        err_mag     (array): Magnitude errors to be converted
        counts      (array): Flux
    Returns:
        flux errors (array): In units of erg/s/cm2/AA
    Reference              : http://classic.sdss.org/dr7/algorithms/fluxcal.html
    """
    return (err_mag * np.log(10) * counts) / 2.5

def plot_original(x, y, c, title = None):
    """ Show original light curves
    Args:
        x     (array): x-axis
        y     (array): y-axis
        c     (array): y errors
        title (str)  : Appropriate plot title
    Returns:
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
    Args:
        x       (array): Flux of the emission-line
        tx      (array): Corresponding observation dates
        y       (array): Flux of the continuum-emission
        ty      (array): Corresponding observation dates
        minlag  (float): Lag range low (days)
        maxlag  (float): Lag range high (days)
        step    (float): Lag bin width (days)
        shifted   (str): Light curve to be shifted, C for continuum and L for line
    Returns:
        lags    (array): Lag values in days
        corr    (array): Cross-correlation values

    Reference          : White & Peterson (1994)
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
    Args:
        lags      (array): Lag values
        corr      (array): Cross-correlation values
        threshold (int)  : Only select values above the specified threshold (80 or 90). For much broader peaks, 90 is preferred
    Returns:
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
    Args:
        lags  (array): Lag values
        corr  (array): cross-correlation values
    Returns:
        Plotted cross-correlation results
    """
    plt.figure()

    plt.plot(lags, corr, '-k')
    plt.ylim([0, 1])
    plt.title(title)
    plt.xlabel('lags [days]', fontsize = 'x-large')
    plt.ylabel('r', fontsize = 'x-large')
    plt.grid()

def show_CCPD(lags, results, title = None, kind = None):
    """ Calculate the upper and lower uncertainties for both the centroid and the peak
    Args:
        results (array): Centroids or peaks
        lags    (array): Lag values
        title   (str)  : Appropriate plot title
        type    (str)  : Centroid or peak
    Returns:
        Print the median and the associated upper and lower uncertainties
        Plotted histogram
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

    print '--------------------{} results -----------------------'.format(kind)
    print 'The upper error: {} days'.format(upper)
    print 'The lower error: {} days'.format(lower)
    print 'The median: {} days'.format(lag)
    
    plt.figure()
    
    plt.hist(results, lags, normed = True, histtype = 'step', color = 'black')
    plt.xlabel('Lags [days]', fontsize = 'x-large')
    plt.ylabel('Probability', fontsize = 'x-large')
    plt.title(title, fontsize = 'x-large')

# Error estimation using flux randomization and random subset selection method

def FR_RSS(x, tx, xerr, y, ty, yerr, minlag, maxlag, step, num = None):
    """ Calculate upper and lower uncertainties using flux randomization and random subset selection (FR/RSS) method
    Args:
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
    Returns:
        Print the median and the associated upper and lower uncertainties
        Plotted histogram
    Reference          : Peterson (1998)
    """
    # Array to store all centroids
    centroids = np.zeros(num, dtype = np.float64)
    
    # Array to store all peaks
    peaks = np.zeros(num, dtype = np.float64)

    for index in range(num):
        # Random subset selection with replacement
        tx_rand_samp = np.random.choice(tx, tx.size, replace = True)
        ty_rand_samp = np.random.choice(ty, ty.size, replace = True)

        # Remove duplicates
        tx_rm_dupl = np.unique(tx_rand_samp)
        ty_rm_dupl = np.unique(ty_rand_samp)

        # Sort
        tx_rm_dupl.sort()
        ty_rm_dupl.sort()

        # Get indices
        x_indices = np.searchsorted(tx, tx_rm_dupl)
        y_indices = np.searchsorted(ty, ty_rm_dupl)
  
        # Get fluxes
        x_rm_dupl = x[x_indices]
        y_rm_dupl = y[y_indices]

        # Get corresponding error fluxes
        xerr_rm_dupl = xerr[x_indices]
        yerr_rm_dupl = yerr[y_indices]

        # Add Gaussian noise
        x_rm_dupl_noise = np.random.normal(x_rm_dupl, xerr_rm_dupl)
        y_rm_dupl_noise = np.random.normal(y_rm_dupl, yerr_rm_dupl)

        # Cross-correlation
        first_pass = cross_corr(x_rm_dupl_noise, tx_rm_dupl, y_rm_dupl_noise, ty_rm_dupl, minlag, maxlag, step, shifted = 'C')
        second_pass = cross_corr(x_rm_dupl_noise, tx_rm_dupl, y_rm_dupl_noise, ty_rm_dupl, minlag, maxlag, step, shifted = 'L')

        avg_corr = np.mean(np.array([first_pass, second_pass]), axis = 0)
        avg_lags = np.arange(minlag, maxlag + step, step)

        
        # Get the lags
        lags = get_lags(avg_lags, avg_corr, threshold = 80)
        
        # Append
        centroids[index] = lags['centroid']
        peaks[index] = lags['peak']

    binned = np.arange(minlag, maxlag + step, step)

    # Show errors and plots
    show_CCPD(binned, centroids, title = r'$\tau_{cent}$', kind = 'Centroid')
    show_CCPD(binned, peaks, title = r'$\tau_{peak}$', kind = 'Peak')


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

    spacing = 1 # day
    tnew_cont = np.arange(cont_t[0], cont_t[-1] + spacing, spacing)
    tnew_line = np.arange(line_t[0], line_t[-1] + spacing, spacing)

    interp_cont = np.interp(tnew_cont, cont_t, cont_f)
    interp_line = np.interp(tnew_line, line_t, line_f) 

    # Lag range
    lags = np.arange(args.lgl, args.lgh + args.dt, args.dt)
    
    # Auto-correlation

    acf_cont = cross_corr(interp_cont, tnew_cont, interp_cont, tnew_cont, args.lgl, args.lgh, args.dt, shifted = 'C')
    acf_line = cross_corr(interp_line, tnew_line, interp_line, tnew_line, args.lgl, args.lgh, args.dt, shifted = 'L')

    show_CCF(lags, acf_cont, title = 'Continuum ACF')
    show_CCF(lags, acf_line, title = 'Line ACF')

    # Cross-correlation

    first_pass = cross_corr(line_f, line_t, cont_f, cont_t, args.lgl, args.lgh, args.dt, shifted = 'C')
    second_pass = cross_corr(line_f, line_t, cont_f, cont_t, args.lgl, args.lgh, args.dt, shifted = 'L')
    
    avg_corr = np.mean(np.array([first_pass, second_pass]), axis = 0)
    
    show_CCF(lags, avg_corr, title = 'CCF')

    orig_lags = get_lags(lags, avg_corr, threshold = 80)

    print 'The centroid of the original data: {} days'.format(orig_lags['centroid'])
    print 'The peak of the original data: {} days'.format(orig_lags['peak'])


    FR_RSS(line_f, line_t, line_err, cont_f, cont_t, cont_err, args.lgl, args.lgh, args.dt, num = 1000)
    plt.show()
