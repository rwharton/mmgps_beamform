import numpy as np
import matplotlib.pyplot as plt 
import glob

from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib import colors, ticker, colormaps
from matplotlib.patches import Circle, Rectangle

from astropy.table import Table
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord

from argparse import ArgumentParser
import sys
import os
import json

from collections import Counter
from scipy.special import erfcinv
from scipy.special import ndtri_exp, log_ndtr


#########################
###  ROC CURVE STUFF  ###
#########################

def PR_idx(xx, bzz, izz):
    """
    """
    sbm = set(bzz)
    sim = set(izz)
    all_src = set(xx)

    # True positives
    tp = len( sbm.intersection(sim) )
    
    # False positives
    fp = len( sbm.difference(sim) ) 

    # False negatives
    fn = len( sim.difference(sbm) )

    prec = tp / (tp + fp)
    rec  = tp / (tp + fn)

    return prec, rec


def get_PR_rates(catfile, bm_cat, im_cat, bsnrs, isnr):
    """
    Get prec and rec
    """
    precs = []
    recs = []
    ns = []

    for bsnr in bsnrs:
        xx, izz = get_filter_rows(catfile, im_cat, 
                                  do_filter=True, snr_min=bsnr)
        _, bzz = get_filter_rows(catfile, bm_cat, 
                                 do_filter=True, snr_min=bsnr)
        prec, rec = PR_idx(xx, bzz, izz)
        precs.append(prec)
        recs.append(rec)
        ns.append( len(bzz) )

    precs = np.array(precs)
    recs = np.array(recs)
    ns = np.array(ns)

    return precs, recs, ns


def ROC_idx(xx, bzz, izz):
    """
    """ 
    # Detections
    y_bm = set(bzz)
    y_im = set(izz)

    # non detections
    n_bm = set(xx).difference(bzz)
    n_im = set(xx).difference(izz)

    # True positives
    tp = len( y_bm.intersection(y_im) )
    
    # False positives
    fp = len( y_bm.intersection(n_im) ) 

    # False negatives
    fn = len( n_bm.intersection(y_im) )

    # True negatives 
    tn = len( n_bm.intersection(n_im) )

    tpr = tp / (tp + fp)
    fpr  = fp / (fp + tn)

    return tpr, fpr


def get_ROC_rates(catfile, bm_cat, im_cat, bsnrs, isnr):
    """
    Get tpr anddddddfpr for bsnrs
    """
    xx, izz = get_filter_rows(catfile, im_cat, 
                              do_filter=True, snr_min=isnr)

    tprs = []
    fprs = []

    for bsnr in bsnrs:
        _, bzz = get_filter_rows(catfile, bm_cat, 
                                 do_filter=True, snr_min=bsnr)
        tpr, fpr = ROC_idx(xx, bzz, izz)
        tprs.append(tpr)
        fprs.append(fpr)

    tprs = np.array(tprs)
    fprs = np.array(fprs)

    return tprs, fprs


##########################
###   RM IMAGE PLOT    ###
##########################

def transform_lim(cc0, size_deg, wcs ):
    """
    Get pixel limits so that the image size
    is size_deg degrees on a side
    """
    ny, nx = wcs.array_shape

    offset = (size_deg/2) * u.deg

    cc_ll = cc0.directional_offset_by(135 * u.deg, offset * np.sqrt(2) )
    cc_ur = cc0.directional_offset_by(-45 * u.deg, offset * np.sqrt(2) )

    pix_ll = wcs.world_to_array_index(cc_ll)
    pix_ur = wcs.world_to_array_index(cc_ur)

    pix_xlim = ( float(pix_ll[1]), float(pix_ur[1]) )
    pix_ylim = ( float(pix_ll[0]), float(pix_ur[0]) )

    return pix_xlim, pix_ylim


def make_plot_rms(fitsfile, catfile, use_nums=None,
                  vrange = None, imsize=None, outbase=None,
                  clog=False, crange=(None, None), bounds=None,
                  nocolor='k', dark=False, fwhm=None):
    """
    make plot
    """
    if outbase is not None:
        plt.ioff()

    if dark:
        plt.style.use('dark_background')

    # Read catalog
    tab = Table.read(catfile)

    # Get coordinates
    ccs = SkyCoord(tab['RA'], tab['DEC'])

    # Read RM catalog data
    rms = tab['RM']
    snrs = tab['PI_SNR']

    if use_nums is not None:
        xx = use_nums
    else:
        xx = np.arange(len(tab))

    ccs = ccs[xx]
    rms = rms[xx]
    snrs = snrs[xx]

    print(f"RM Range:  {np.min(rms)}, {np.max(rms)}")

    # Setup figure and axis
    hdu = fits.open(fitsfile)[0]
    wcs = WCS(hdu.header)
    cc0 = SkyCoord( hdu.header['CRVAL1'], hdu.header['CRVAL2'], unit=u.deg, frame='fk5')

    # Assuming we also have freq and stokes axis
    dat = hdu.data[0, 0, :, :]
    subwcs = wcs[0, 0, :, :]
    #dat = hdu.data[:, :]
    #subwcs = wcs[:, :]

    fig, ax = plt.subplots(subplot_kw=dict(projection=subwcs))
    fig.set_figheight(8)
    fig.set_figwidth(12)

    # Get image data
    cmap = 'bone'
    if vrange is None:
        #vrange = (np.nanmin(dat), np.nanmax(dat))
        med, sig, src_max, vlo, vhi = get_cut_stats(dat, frac=0.9)
        vrange = ( med - 4 * sig, med + 10 * sig )
    
    ax.imshow(dat, vmin=vrange[0], vmax=vrange[1],
              origin='lower', cmap=cmap,
              interpolation='nearest')
    lcol = 'k'
    if dark:
        lab_col = 'w'
        tick_col = 'w'
        fig.set_facecolor('k')
        fig.set_edgecolor('k')
        #ax.set_frame_on(False)
        for key, spine in ax.spines.items():
            spine.set_color('w')
    if cmap == 'bone':
        tick_col = 'w'
        lab_col = 'k'

    ax.set_xlabel('RA', fontsize=18, color=lab_col)
    ax.set_ylabel('DEC', fontsize=18, color=lab_col)
    ax.tick_params(axis='both', labelcolor=lab_col, color=tick_col,
                   labelsize=14, direction="in", length=10)

    #ax.coords[0].set_major_formatter('d.dd')
    #ax.coords[1].set_major_formatter('d.dd')

    ax.set_facecolor('k')

    mcol = None

    lstr = '${\\rm RM}~({\\rm rad~m}^{-2})$'
    col_cmap = 'coolwarm'

    val_min = np.min(rms)
    val_max = np.max(rms)
    val_abs_max = np.max(np.abs(rms))

    if crange[0] is None:
        cmin = -1 * val_abs_max
    else:
        cmin = crange[0]

    if crange[-1] is None:
        cmax = +1 * val_abs_max
    else:
        cmax = crange[-1]

    if bounds is None:
        norm = colors.Normalize(vmin=cmin, vmax=cmax)
    else:
        cmin = bounds[0]
        cmax = bounds[-1]
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    if (val_min < cmin) & (val_max > cmax):
        extend = 'both'
    elif (val_min < cmin) & (val_max <= cmax):
        extend = 'min'
    elif (val_min >= cmin) & (val_max > cmax):
        extend = 'max'
    else:
        extend = 'neither'
    ccm = colormaps.get_cmap(col_cmap)
    sizes = 100 * (snrs / 10)
    ax.scatter(ccs.ra, ccs.dec,
               transform=ax.get_transform('fk5'),
               marker='o', s=sizes, lw=4,
               facecolor='None', edgecolor=ccm(norm(rms)))

    toff = 2 * u.arcmin
    if dark or (cmap=='bone'):
        tcol = 'Yellow'
    else:
        tcol = '0.2'
    for ii, cc in enumerate(ccs):
        cct = cc.directional_offset_by(180 * u.deg, toff)
        ax.text_coord(cct, str(xx[ii]), ha='center', va='top',
                      size=10, c=tcol)

    # ugly hack
    cax = ax.scatter(ccs.ra, ccs.dec,
                     transform=ax.get_transform('fk5'),
                     marker='o', s=0, c=rms, cmap=ccm, norm=norm)

    cbar = plt.colorbar(cax, shrink=0.8, extend=extend)
    cbar.ax.set_ylabel(lstr)

    # primary beam if desired
    if fwhm is not None:
        r = SphericalCircle((cc0.ra, cc0.dec), 0.5 * fwhm * u.degree,
                     edgecolor='white', facecolor='none', lw=2, ls='--',
                     transform=ax.get_transform('fk5'))
        ax.add_patch(r)

    if imsize is not None:
        xlim, ylim = transform_lim(cc0, imsize, subwcs )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if outbase is not None:
        plt.savefig(f"{outbase}.png", dpi=150, bbox_inches='tight')
        plt.close()
        plt.ion()
    else:
        plt.show()

    if dark:
        plt.style.use('default')
    return



##########################
###  RM COMPARE STUFF  ###
##########################

def get_sig(n, snr):
    """
    Get equivalent normal sigma from rayleigh
    snr and n independent samples
    """
    lnx = np.log(n) - 0.5 * snr**2
    sig_N = -1 * ndtri_exp(lnx)
    return sig_N


def sig_to_snr(n, sig):
    """
    get the snr for norm equiv sigma
    """
    snr = np.sqrt( 2 * (np.log(n) - log_ndtr(-1 * sig) ) )
    return snr


def isl_filter(fits_file, max_comp=1, min_flux=0, userows=[]):
    """
    Set a maximum number of components a given
    'island' can have.  These are clusters of
    sources that are close to one another. While
    real component certainly exist (e.g. lobes
    or resolved structure), these can often be
    artifacts near bright sources

    If min_flux > 0, then we only consider islands
    with the brightest component brigher than min_flux.
    Other islands are not filtered.  This can be useful
    if you mainly want to focus on artifacts.
    """
    full_tab = Table.read(fits_file)

    if len(userows):
        tab = full_tab[userows]
    else:
        tab = full_tab

    isl_ids = tab['Isl_id']
    flux = tab['Total_flux']

    isl_count = Counter(isl_ids)

    idx_bad = []
    for isl_id, isl_num in isl_count.items():
        if isl_num <= max_comp:
            continue

        # get row nums of sources in that isl
        xx = np.where(isl_ids == isl_id)[0]

        # get fluxes, skip if max flux is less than
        # the min flux
        flux_isl = flux[xx]
        if np.max(flux[xx]) <= min_flux:
            continue

        # sort by descending flux
        yy = np.argsort(flux_isl)[::-1]

        # Get the indices beyond max_comp that
        # we want to remove
        xx_bad = xx[ yy[max_comp:] ]

        idx_bad += xx_bad.tolist()

    idx_good = np.setdiff1d( np.arange(len(tab)), np.array(idx_bad) )

    if len(userows):
        xx_out = userows[idx_good]
    else:
        xx_out = idx_good

    return xx_out


def get_filter_rows(catfile, do_filter=True, snr_min=0):
    """
    Do island filter and snr cut
    """
    # Read table
    tab = Table.read(catfile)
    snrs = tab['PI_SNR']

    if do_filter:
        xx = isl_filter(catfile)
    else:
        xx = np.arange(len(tab))
    
    yy = np.where(snrs[xx] >= snr_min)[0]

    zz = xx[yy]

    return xx, zz


def make_pol_frac_plot(catfile, outbase=None,
                       mark_snr=None,
                       do_filter=True,
                       mark_nums=None, title=None):
    """
    Make a plot of total intesity flux vs pol frac
    Show lines of nsigma * noise pol frac
    """
    if outbase is not None:
        plt.ioff()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # get flux from source catalog
    tab = Table.read(catfile)
    Iflux = tab['Total_flux'].value * 1e3 # mJy
    PIs = tab['PI'].value * 1e3 # mJy
    pi_errs = tab['E_PI'].value * 1e3 # mJy
    snrs = tab['PI_SNR'].value

    xall = np.arange(len(tab))
    if do_filter:
        xx  = isl_filter(catfile)
        xxc = np.setdiff1d(xall, xx)
    else:
        xx = xall

    noise = np.median(pi_errs[xx])

    pfrac = PIs / Iflux

    ax.plot(Iflux[xx], 100 * pfrac[xx], marker='o', c='k', ls='', ms=4)

    if do_filter:
        ax.plot(Iflux[xxc], 100 * pfrac[xxc], marker='o', c='0.6', ls='', ms=1)

    nsigs = np.array([5, 10, 15])

    x = np.linspace( np.min(Iflux[xx]), np.max(Iflux[xx]), 1000)
    for ii, nn in enumerate(nsigs):
        label = f"${{{nn}}}\\sigma$"
        nsnr = sig_to_snr(800, nn)
        ax.plot(x, 100 * nsnr * noise / x, lw=1 + 0.5 * (3-ii), ls='--', label=label)

    if mark_nums is not None:
        ax.plot(Iflux[mark_nums], 100 * pfrac[mark_nums], marker='o',
                mec='r', mfc='none', ls='', ms=10)

    if mark_snr is not None:
        yy = np.where(snrs[xx] >= mark_snr)[0]
        ax.plot(Iflux[xx[yy]], 100 * pfrac[xx[yy]], marker='o',
                mec='r', mfc='none', ls='', ms=10)

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x),0)))).format(x)))

    ax.set_xlabel("Total Intensity Flux Density (mJy)", fontsize=16)
    ax.set_ylabel("Polarisation Fraction (%)", fontsize=16)

    plt.legend(loc='lower left', fontsize=16)

    if title is not None:
        plt.title(title, fontsize=16)

    if outbase is not None:
        plt.savefig(f"{outbase}.png", dpi=150, bbox_inches='tight')
        plt.close()
        plt.ion()

    else:
        plt.show()

    if mark_snr:
        zz = xx[yy]
    else:
        zz = xx

    return zz


############################
###  IMAGE CUTOUT STUFF  ###
############################

def get_all_ccs(catfile):
    """
    Get list of sky coord objects from catfile table
    """
    tab = Table.read(catfile)
    ccs = SkyCoord(tab['RA'], tab['DEC'])
    return ccs

def get_beam_centers(cc_list, wcs):
    """
    Get beam centers in array indices
    """
    idx_arr = np.zeros( shape=(len(cc_list), 2) )
    for ii, cc in enumerate(cc_list):
        # Get row and column index for coordinate
        r0, c0 = wcs.world_to_array_index(cc)
        idx_arr[ii] = [r0, c0]
    return idx_arr

def find_beams_in_field(idx_arr, idx0, size, rpix):
    """
    Look for beams that fall within the field of view
    centered on idx0 with side length size

    assume the beam has radiux rpix

    return array of center locs and ids
    """
    rs = idx_arr[:, 0]
    cs = idx_arr[:, 1]

    cond1 = np.abs( rs - idx0[0] ) < (size // 2) + rpix
    cond2 = np.abs( cs - idx0[1] ) < (size // 2) + rpix

    xx = np.where( cond1 & cond2 )[0]

    idx_offset = idx_arr[xx] - idx0

    return idx_offset, xx


def get_offsets(cc1, cc0, unit='arcmin', frame='fk5'):
    """
    calculate the offsets between two coordinates
    """

    if frame == 'fk5':
        cc1_ra  = SkyCoord(cc1.ra, cc0.dec, frame='fk5')
        cc1_dec = SkyCoord(cc0.ra, cc1.dec, frame='fk5')

        ra_off  = cc1_ra.separation(cc0).arcmin
        dec_off = cc1_dec.separation(cc0).arcmin

        if unit == 'deg':
            fac = 60.
        elif unit == 'arcsec':
            fac = 1/60.
        else:
            fac = 1.

        ra_off /= fac
        dec_off /= fac

        if cc1.ra < cc0.ra:
            ra_off *= -1
        if cc1.dec < cc0.dec:
            dec_off *= -1

        lon_off = ra_off
        lat_off = dec_off

    elif frame == 'galactic':
        cc1_l  = SkyCoord(cc1.galactic.l, cc0.galactic.b, frame=frame)
        cc1_b  = SkyCoord(cc0.galactic.l, cc1.galactic.b, frame=frame)

        l_off  = cc1_l.separation(cc0).arcmin
        b_off  = cc1_b.separation(cc0).arcmin

        if unit == 'deg':
            fac = 60.
        elif unit == 'arcsec':
            fac = 1/60.
        else:
            fac = 1.

        l_off /= fac
        b_off /= fac

        if cc1.galactic.l < cc0.galactic.l:
            l_off *= -1
        if cc1.galactic.b < cc0.galactic.b:
            b_off *= -1

        lon_off = l_off
        lat_off = b_off

    return (lon_off, lat_off)


def get_cutout(cc, data, wcs, size):
    """
    Get the subarray of data centered on the
    SkyCoord cc using the WCS wcs

    size is the size in pixels
    """
    # Get row and column index for coordinate
    r0, c0 = wcs.world_to_array_index(cc)

    cond1 = (r0 >= 0) and (r0 < data.shape[0])
    cond2 = (c0 >= 0) and (c0 < data.shape[1])

    if (cond1 and cond2):

        cdat = data[ r0 - size//2 : r0 + size//2,
                     c0 - size//2 : c0 + size//2 ]
    else:
        cdat = np.array([])

    return cdat, (r0, c0)


def get_cut_stats(cdat, frac=0.9):
    """
    find max at central npix x npix of cdat
    """
    all_cvals = np.sort(cdat.ravel())
    xx = np.where( np.isnan(all_cvals) == False )[0]
    cvals = all_cvals[xx]
    N = len(cvals)
    xlo = int( N * 0.5 * (1-frac) )
    xhi = int( N * 0.5 * (1+frac) )
    cmids = cvals[ xlo : xhi ]

    sig = np.std(cmids)
    vlo = cmids[0]
    vhi = cmids[-1]

    med = np.median(cmids)

    # get src max (roughly)
    nr, nc = cdat.shape
    nr_mid = nr // 2
    nc_mid = nc // 2
    src_max = np.max( cdat[ nr_mid - 3 : nr_mid + 3, nc_mid - 3 : nc_mid + 3 ] )

    return med, sig, src_max, vlo, vhi


def add_cutout_ax(fig, nrows, ncols, index,
                  cdat, dx_deg, dy_deg, vmin=None, vmax=None,
                  title=None, idx_off=None, xx=None,
                  radius=None, bunit='', cmap=None, panel_label=None):
    """
    Make a nice cutout image
    """
    ax = fig.add_subplot(nrows, ncols, index)

    med, sig, src_max, vlo, vhi = get_cut_stats(cdat, frac=0.90)
    if vmin is None:
        vmin = med - 3 * sig #vlo
    if vmax is None:
        vmax = max(0.9 * src_max, med + 6 * sig) #vhi

    if vmax/sig > 30:
        cnorm = colors.PowerNorm(0.5, vmin, vmax)
    else:
        cnorm = colors.Normalize(vmin, vmax)

    cx, cy = cdat.shape
    l_arc = -1 * (cx//2) * dx_deg * 3600
    r_arc = +1 * (cx//2) * dx_deg * 3600

    b_arc = -1 * (cy//2) * dy_deg * 3600
    t_arc = +1 * (cy//2) * dy_deg * 3600

    ext = [l_arc, r_arc, b_arc, t_arc]

    im = ax.imshow(cdat, aspect='equal', origin='lower',
                   interpolation='nearest',
                   extent=ext, cmap=cmap, norm=cnorm)
    """
    if (idx_off is not None) and (len(idx_off)):
        rxy = (ext[0], ext[2])
        rw = -1 * np.abs(ext[1] - ext[0])
        rh = np.abs(ext[3] - ext[2])
        for ii, idx in enumerate(idx_off):
            y = idx[0] * dy_deg * 3600
            x = idx[1] * dx_deg * 3600
            if (x == 0) and (y == 0):
                ls = '-'
            else:
                ls = '--'
            p = Circle((x, y), radius, fc='none', ec='w', lw=2, ls=ls)
            ax.add_artist(p)
            #ax.text(x, y + 1.25 * radius, f"{xx[ii]:03d}",
            #        color='w', ha='center', va='center',
            #        fontsize=12)
    """

    if (idx_off is not None) and (len(idx_off)):
        rxy = (ext[0], ext[2])
        rw = -1 * np.abs(ext[1] - ext[0])
        rh = np.abs(ext[3] - ext[2])
        for ii, idx in enumerate(idx_off):
            y = idx[0] * dy_deg * 3600
            x = idx[1] * dx_deg * 3600
            if (x == 0) and (y == 0):
                ls = '-'
            else:
                ls = '--'

            x1 = y + 10
            y1 = y + 10
            ax.errorbar(x1, y, xerr=5, yerr=0, color='w', marker='none', ms=8, lw=2)
            ax.errorbar(x, y1, xerr=0, yerr=5, color='w', marker='none', ms=8, lw=2)

            #p = Circle((x, y), radius, fc='none', ec='w', lw=2, ls=ls)
            #ax.add_artist(p)


    if panel_label is not None:
        #bbox_fmt = dict(boxstyle="square", ec='1', fc='0.95',
        #                mutation_aspect=0.5)
        bbox_fmt = {}
        ax.text(0.875, 0.875, panel_label, fontsize=40, color='w',
                #bbox=bbox_fmt,
                va='center', ha='center', transform=ax.transAxes)

    ax.set_xlabel("Offset (arcsec)", fontsize=18)
    ax.set_ylabel("Offset (arcsec)", fontsize=18)

    tick_color = 'w'
    tick_label_size=14
    ax.tick_params(axis='x', color=tick_color,
                   labelsize=tick_label_size, direction="in",
                   length=10, labelbottom=True, labeltop=False,
                   bottom=True, top=True)

    ax.tick_params(axis='y', color=tick_color,
                   labelsize=tick_label_size, direction="in",
                   length=10, labelright=False, labelleft=True,
                   left=True, right=True)

    if title is not None:
        ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, shrink=1.0, pad=0.01)
    cbar.set_label(bunit, fontsize=14, rotation=270,
                   labelpad=14, va='center')
    cbar.ax.tick_params(labelsize=14)

    return ax, im


def plot_crosshair2(ax, cc0, offset=10, csize=10, c='w', lw=2):
    """
    Add a crosshair to plot in axis ax near cc0
    but offset by "offset" arcsec
    """
    cc_u_lo = cc0.directional_offset_by(0 * u.deg, offset * u.arcsec)
    cc_u_hi = cc0.directional_offset_by(0 * u.deg, (offset+csize) * u.arcsec)

    cc_l_lo = cc0.directional_offset_by(90 * u.deg, offset * u.arcsec)
    cc_l_hi = cc0.directional_offset_by(90 * u.deg, (offset+csize) * u.arcsec)

    N_lons = [ cc_u_lo.l, cc_u_hi.l ]
    N_lats = [ cc_u_lo.b, cc_u_hi.b ]

    E_lons = [ cc_l_lo.l, cc_l_hi.l ]
    E_lats = [ cc_l_lo.b, cc_l_hi.b ]

    ax.errorbar(cc_u_lo.l, cc_u_lo.b,
                xerr=0 * u.arcsec,
                yerr=csize * u.arcsec,
                transform=ax.get_transform('galactic'),
                color=c, marker='none',
                ms=8, lw=2)

    ax.errorbar(cc_l_lo.l, cc_l_lo.b,
                xerr=csize * u.arcsec,
                yerr=0 * u.arcsec,
                transform=ax.get_transform('galactic'),
                color=c, marker='none',
                ms=8, lw=2)

    return



#####################################
###   QU Spectra + RM Synthesis   ###
#####################################

def read_json(json_file):
    """
    read in json file results for clean and return dict
    """
    with open(json_file, "r") as fin:
        dd = json.load(fin)
    return dd


def read_FDF(datfile, scale=1e6):
    """
    Read in faraday depth spectra

    Data in form (phi_rad_m2, real spec, imag spec)
    """
    dat = np.loadtxt(datfile)
    phis = dat[:, 0]
    r_spec = dat[:, 1] * scale
    i_spec = dat[:, 2] * scale
    m_spec = np.sqrt(r_spec**2 + i_spec**2)
    return phis, r_spec, i_spec, m_spec


def read_IQU(datfile, scale=1e6):
    """
    Read in IQU spectra
    """
    dat = np.loadtxt(datfile)
    freqs  = dat[:, 0]
    I = dat[:, 1] * scale
    Q = dat[:, 2] * scale
    U = dat[:, 3] * scale
    return freqs, I, Q, U 


def get_freq_groups(freqs, min_dfact=3):
    """
    find gaps
    """
    df = np.median( np.diff(freqs) )
    xx = np.where( np.diff(freqs) > min_dfact * df )[0]
    starts = []
    stops = []
    start = 0 
    for jj in xx:
        stop = jj
        starts.append(start)
        stops.append(stop)
        start = stop + 1 

    starts.append(start)
    stops.append(len(freqs))

    starts = np.array(starts)
    stops = np.array(stops)
    
    return starts, stops 
         

def make_QU_plot(basename, xlim=None,
                 title=None, outfile=None):
    """
    Make a nice plot of clean FDF
    
    mark known rm at `rm` if not none
    """
    if outfile is not None:
        plt.ioff()

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)

    datfile = f"{basename}.txt"
    freqs, I, Q, U = read_IQU(datfile, scale=1e3)
    P = np.sqrt(Q**2 + U**2)

    freqs_MHz = freqs / 1e6
    lam2s = (3e8 / freqs)**2

    starts, stops = get_freq_groups(freqs_MHz)

    Qcol = 'tab:blue'
    Ucol = 'tab:orange'
    Pcol = 'k'
    for ii in range(len(starts)):
        if ii == 0:
            Qlab = 'Q'
            Ulab = 'U'
            Plab = 'L'
        else:
            Qlab = Ulab = Plab = None
        xx = slice(starts[ii], stops[ii])
        ax.plot(lam2s[xx], Q[xx], c=Qcol, label=Qlab, marker='')
        ax.plot(lam2s[xx], U[xx], c=Ucol, label=Ulab, marker='')
        #ax.plot(lam2s[xx], P[xx], c=Pcol, label=Plab, marker='', alpha=0.3)

    ax.set_xlabel("${\\lambda}^2~({\\rm m}^{2})$", 
                  fontsize=14)
    ax.set_ylabel("${\\rm Flux~Density}~({\\rm mJy~beam}^{-1})$", 
                  fontsize=14)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    
    plt.tick_params(axis='both', which='major', 
                    direction='in', length=7, 
                    top=True, bottom=True, 
                    left=True, right=True, 
                    labelsize=12)
    
    plt.tick_params(axis='both', which='minor', 
                    direction='in', length=4, 
                    top=True, bottom=True, 
                    left=True, right=True, 
                    labelsize=12)

    plt.grid(alpha=0.2)

    plt.legend(fontsize=12)

    if title is not None:
        plt.title(title, fontsize=16)

    if outfile is not None:
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        plt.close()
        plt.ion()
    else:
        plt.show()

    return


def make_clean_plot(basename, xlim=None,
                    title=None, outfile=None):
    """
    Make a nice plot of clean FDF
    
    mark known rm at `rm` if not none
    """
    if outfile is not None:
        plt.ioff()

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)

    dclean = f"{basename}_FDFclean.dat"
    ddirty = f"{basename}_FDFdirty.dat"
    dmodel = f"{basename}_FDFmodel.dat"

    phis, r_c, i_c, m_c = read_FDF(dclean, scale=1e3)
    _   , r_d, i_d, m_d = read_FDF(ddirty, scale=1e3)
    _   , r_m, i_m, m_m = read_FDF(dmodel, scale=1e3)

    ax.plot(phis, m_c, lw=1., c='k', label="CLEAN")
    ax.plot(phis, m_d, lw=1, c='k', alpha=0.2, label="Dirty")

    # Add clean model components
    xx = np.where( m_m )[0]
    if len(xx):
        for jj, ii in enumerate(xx):
            if jj == 0:
                label = "Model"
            else:
                label = None
            ax.plot([phis[ii]] * 2 , [0, m_m[ii]], lw=1, c='r', label=label)
    

    ax.set_xlabel("$\\phi~({\\rm rad~m}^{-2})$", fontsize=16)
    ax.set_ylabel("${\\rm Flux~Density}~({\\rm mJy~beam}^{-1})$", 
                  fontsize=14)

    max_val = np.max(m_c)
    ax.set_ylim(-0.1 * max_val, 1.2 * max_val)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    
    plt.tick_params(axis='both', which='major', 
                    direction='in', length=7, 
                    top=True, bottom=True, 
                    left=True, right=True, 
                    labelsize=12)
    
    plt.tick_params(axis='both', which='minor', 
                    direction='in', length=4, 
                    top=True, bottom=True, 
                    left=True, right=True, 
                    labelsize=12)

    plt.grid(alpha=0.2)

    plt.legend(fontsize=12)

    if title is not None:
        plt.title(title, fontsize=16)

    if outfile is not None:
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        plt.close()
        plt.ion()
    else:
        plt.show()

    return


###########################
###   SUB PANEL PLOTS   ###
###########################

def add_cutout_ax(fig, nrows, ncols, index,
                  cdat, dx_deg, dy_deg, vmin=None, vmax=None,
                  title=None, idx_off=None, xx=None,
                  radius=None, bunit='', cmap=None, panel_label=None):
    """
    Make a nice cutout image
    """
    ax = fig.add_subplot(nrows, ncols, index)

    med, sig, src_max, vlo, vhi = get_cut_stats(cdat, frac=0.90)
    if vmin is None:
        vmin = med - 3 * sig #vlo
    if vmax is None:
        vmax = max(0.9 * src_max, med + 6 * sig) #vhi

    if vmax/sig > 30:
        cnorm = colors.PowerNorm(0.5, vmin, vmax)
    else:
        cnorm = colors.Normalize(vmin, vmax)

    cx, cy = cdat.shape
    l_arc = -1 * (cx//2) * dx_deg * 3600
    r_arc = +1 * (cx//2) * dx_deg * 3600

    b_arc = -1 * (cy//2) * dy_deg * 3600
    t_arc = +1 * (cy//2) * dy_deg * 3600

    ext = [l_arc, r_arc, b_arc, t_arc]

    im = ax.imshow(cdat, aspect='equal', origin='lower',
                   interpolation='nearest',
                   extent=ext, cmap=cmap, norm=cnorm)

    if (idx_off is not None) and (len(idx_off)):
        rxy = (ext[0], ext[2])
        rw = -1 * np.abs(ext[1] - ext[0])
        rh = np.abs(ext[3] - ext[2])
        for ii, idx in enumerate(idx_off):
            y = idx[0] * dy_deg * 3600
            x = idx[1] * dx_deg * 3600
            if (x == 0) and (y == 0):
                ls = '-'
            else:
                ls = '--'

            x1 = y + 10
            y1 = y + 10
            ax.errorbar(x1, y, xerr=5, yerr=0, color='w', marker='none', ms=8, lw=2)
            ax.errorbar(x, y1, xerr=0, yerr=5, color='w', marker='none', ms=8, lw=2)


    if panel_label is not None:
        bbox_fmt = {}
        ax.text(0.875, 0.875, panel_label, fontsize=40, color='w',
                #bbox=bbox_fmt,
                va='center', ha='center', transform=ax.transAxes)

    ax.set_xlabel("Offset (arcsec)", fontsize=18)
    ax.set_ylabel("Offset (arcsec)", fontsize=18)

    tick_color = 'w'
    tick_label_size=14
    ax.tick_params(axis='x', color=tick_color,
                   labelsize=tick_label_size, direction="in",
                   length=10, labelbottom=True, labeltop=False,
                   bottom=True, top=True)

    ax.tick_params(axis='y', color=tick_color,
                   labelsize=tick_label_size, direction="in",
                   length=10, labelright=False, labelleft=True,
                   left=True, right=True)

    if title is not None:
        ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, shrink=1.0, pad=0.01)
    cbar.set_label(bunit, fontsize=14, rotation=270,
                   labelpad=14, va='center')
    cbar.ax.tick_params(labelsize=14)

    return ax, im


def add_clean_ax(fig, nrows, ncols, index, 
                 basename, phi_win_nrmsf=None, snr=10, 
                 title=None, outfile=None):
    """
    Make a nice plot of clean FDF
    
    mark known rm at `rm` if not none
    """
    ax = fig.add_subplot(nrows, ncols, index)

    dclean = f"{basename}_FDFclean.dat"
    ddirty = f"{basename}_FDFdirty.dat"
    dmodel = f"{basename}_FDFmodel.dat"
    jsynth = f"{basename}_RMsynth.json"
    jclean = f"{basename}_RMclean.json"

    phis, r_c, i_c, m_c = read_FDF(dclean, scale=1e3)
    _   , r_d, i_d, m_d = read_FDF(ddirty, scale=1e3)
    _   , r_m, i_m, m_m = read_FDF(dmodel, scale=1e3)

    # Read SYNTH json file to get RMSF fwhm
    sjd = read_json(jsynth)
    RMSF_fwhm = sjd['fwhmRMSF'] 

    # Read CLEAN json file, get relevant values 
    jd = read_json(jclean)
    dFDF = jd['dFDFcorMAD'] * 1e3
    phi_pk = jd['phiPeakPIfit_rm2']

    ax.plot(phis, m_c, lw=1., c='k', label="CLEAN")
    ax.plot(phis, m_d, lw=1, c='k', alpha=0.2, label="Dirty")

    # Add clean model components
    xx = np.where( m_m )[0]
    if len(xx):
        for jj, ii in enumerate(xx):
            if jj == 0:
                label = "Model"
            else:
                label = None
            ax.plot([phis[ii]] * 2 , [0, m_m[ii]], lw=1, c='r', label=label)

    if snr > 0:
        ax.axhline(y=snr * dFDF, ls='--', c='Orange', alpha=0.5)
    

    ax.set_xlabel("$\\phi~({\\rm rad~m}^{-2})$", fontsize=16)
    ax.set_ylabel("${\\rm Flux~Density}~({\\rm mJy~beam}^{-1})$", 
                  fontsize=14)

    ax.text(0.1, 0.9, f"$\\phi_{{\\rm pk}} = {phi_pk:+.1f}~{{\\rm rad~m}}^{{-2}}$", 
            transform=ax.transAxes, ha='left', fontsize=16)

    max_val = np.max(m_c)
    if (snr > 0) and ( snr * dFDF > max_val ):
        max_val = snr * dFDF
    ax.set_ylim(-0.1 * max_val, 1.2 * max_val)

    if phi_win_nrmsf is not None:
        phi_win = RMSF_fwhm * phi_win_nrmsf
        xlim = (phi_pk - phi_win, phi_pk + phi_win)
        ax.set_xlim(xlim)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    
    ax.tick_params(axis='both', which='major', 
                    direction='in', length=7, 
                    top=True, bottom=True, 
                    left=True, right=True, 
                    labelsize=12)
    
    ax.tick_params(axis='both', which='minor', 
                    direction='in', length=4, 
                    top=True, bottom=True, 
                    left=True, right=True, 
                    labelsize=12)

    ax.grid(alpha=0.2)

    ax.legend(fontsize=12, loc='upper right')

    return ax 


def add_QU_ax(fig, nrows, ncols, index, 
                 basename, xlim=None,
                 title=None, outfile=None):
    """
    Make a nice plot of clean FDF
    
    mark known rm at `rm` if not none
    """
    ax = fig.add_subplot(nrows, ncols, index)

    datfile = f"{basename}.txt"
    freqs, I, Q, U = read_IQU(datfile, scale=1e3)
    P = np.sqrt(Q**2 + U**2)

    freqs_MHz = freqs / 1e6
    lam2s = (3e8 / freqs)**2

    starts, stops = get_freq_groups(freqs_MHz)

    Qcol = 'tab:blue'
    Ucol = 'tab:orange'
    Pcol = 'k'
    for ii in range(len(starts)):
        if ii == 0:
            Qlab = 'Q'
            Ulab = 'U'
            Plab = 'L'
        else:
            Qlab = Ulab = Plab = None
        xx = slice(starts[ii], stops[ii])
        ax.plot(lam2s[xx], Q[xx], c=Qcol, label=Qlab, marker='')
        ax.plot(lam2s[xx], U[xx], c=Ucol, label=Ulab, marker='')
        #ax.plot(lam2s[xx], P[xx], c=Pcol, label=Plab, marker='', alpha=0.3)

    ax.set_xlabel("${\\lambda}^2~({\\rm m}^{2})$", 
                  fontsize=14)
    ax.set_ylabel("${\\rm Flux~Density}~({\\rm mJy~beam}^{-1})$", 
                  fontsize=14)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    
    ax.tick_params(axis='both', which='major', 
                    direction='in', length=7, 
                    top=True, bottom=True, 
                    left=True, right=True, 
                    labelsize=12)
    
    ax.tick_params(axis='both', which='minor', 
                    direction='in', length=4, 
                    top=True, bottom=True, 
                    left=True, right=True, 
                    labelsize=12)

    ax.grid(alpha=0.2)

    ax.legend(fontsize=12)

    return ax


#######################
### COMBINE IT ALL  ###
#######################

def make_spec_cutouts(catfile, fits_file, rm_spec_dir, 
                      bstr = "beam{:05d}", 
                      outbase='src', name_dict=None,
                      size_arcsec=120, radius_arcsec=20, frac=0.9,
                      vmin=None, vmax=None, frame='fk5',
                      scale_fac=1e3, cmap='inferno', use_rows=None, 
                      phi_win_nrmsf=10, snr=10):
    """
    Make cutouts from coord list cc_list
    """
    plt.ioff()
    #outtype = "pdf"
    outtype = "png"

    # Read catalog
    #freqs, freq_MFS = get_freqs(catfile)
    tab = Table.read(catfile)

    # Get coordinates
    ccs = SkyCoord(tab['RA'], tab['DEC'])

    # if use_rows not specified, use them all
    if use_rows is None:
        use_rows = range(len(tab))

    # Now get stuff ready for cutout
    hdulist = fits.open(fits_file)
    hdu = hdulist[0]

    if hdu.header.get('CDELT1') is not None:
        cdelt1 = hdu.header['CDELT1']
        cdelt2 = hdu.header['CDELT2']

    else:
        cdelt1 = hdu.header['CD1_1']
        cdelt2 = hdu.header['CD2_2']
    # Get units if possible
    bunit = ''
    if hdu.header.get('BUNIT') is not None:
        bunit = hdu.header.get('BUNIT')
        bunit.strip()

    if (scale_fac == 1000) and (bunit != ''):
        bunit = 'm' + bunit

    if bunit.lower() == "mjy/beam":
        bunit = "${\\rm mJy~beam}^{-1}$"

    wcs = WCS(hdu.header)

    # Assuming we also have freq and stokes axis
    dat = hdu.data[0, 0, :, :]
    subwcs = wcs[0, 0, :, :]

    # Get center coord
    r0, c0 = subwcs.array_shape
    cc0 = subwcs.array_index_to_world(r0//2 + 1, c0//2 + 1 )

    idx_arr = get_beam_centers(ccs, subwcs)

    size = int( size_arcsec / (np.abs(cdelt1) * 3600) )
    rpix = int( radius_arcsec / (np.abs(cdelt1) * 3600) )

    for jj, ii in enumerate(use_rows):
        print(f"\rCand {jj+1} / {len(use_rows)}", end="")
        cc = ccs[ii]
        if name_dict is not None:
            src_name = name_dict[ii]
        else:
            src_name = f"{outbase}_{ii:03d}"

        bname = f"{outbase}_{ii:03d}"
        outname = f"{bname}_cutout.{outtype}"


        ###############################
        #####  GET CUTOUT STUFF   #####
        ###############################

        lon_off, lat_off = get_offsets(cc, cc0, unit='arcmin', frame=frame)
        cdat, idx0 = get_cutout(cc, dat, subwcs, size)
        if cdat.size == 0:
            continue

        search_size = 1
        rpix = 1
        idx_offset, xx = find_beams_in_field( idx_arr, idx0,
                                              search_size, rpix)
        #title = bname + ": "
        #"""
        title = ""
        if frame == 'fk5':
            title += f"$(\\Delta \\alpha, \\Delta \\delta) = ({lon_off:+.1f}^{{\\prime}}, {lat_off:+.1f}^{{\\prime}})$"
        else:
            ell = cc.galactic.l.value
            b   = cc.galactic.b.value
            title += f"$(\\ell = {ell:.2f}^{{\\circ}}, b={b:.2f}^{{\\circ}}) $"
        #"""
        #title = None

        #########################
        #####  MAKE FIGURE  #####
        #########################

        # make figure
        fig = plt.figure(figsize=(14, 12))

        # Add cutout
        panel_label = None
        ax_im, im = add_cutout_ax(fig, 2, 2, 1,
                                  cdat * scale_fac, cdelt1, cdelt2,
                                  idx_off=idx_offset, xx=xx,
                                  radius=radius_arcsec,
                                  vmin=vmin, vmax=vmax,
                                  title=title,
                                  bunit=bunit, cmap=cmap,
                                  panel_label=panel_label)

        #basename = f"{rm_spec_dir}/beam{ii:05d}_full/beam{ii:05d}_full"
        bname = bstr.format(ii)
        basename = f"{rm_spec_dir}/{bname}/{bname}"

        # add rm clean spec
        ax_clean = add_clean_ax(fig, 2, 2, 2, basename, snr=snr)


        # add qu plot
        ax_qu = add_QU_ax(fig, 2, 2, 3, basename)

        # add rm clean zoom spec
        ax_clean = add_clean_ax(fig, 2, 2, 4, basename, 
                                phi_win_nrmsf=phi_win_nrmsf, snr=snr)

        plt.savefig(outname, dpi=150, bbox_inches="tight")
        plt.close()

    hdulist.close()
    plt.ion()
    return




def parse_input():
    """
    Parse arguments to plot_beam
    """
    prog_desc = "Make RM Diagnostic Plots"
    parser = ArgumentParser(description=prog_desc)

    parser.add_argument('--outdir',
                        help='Output directory for plots (def: cwd)',
                        default='.', required=False)
    parser.add_argument('--catfile',
                        help='FITS catalog of beam sources',
                        required=True)
    parser.add_argument('--imfile',
                        help='FITS Image of field',
                        required=True)
    parser.add_argument('--snr_cut', type=float,
                        help='Minimum PI SNR',
                        default=1.0, required=False)
    parser.add_argument('--outbase',
                        help='Base of output files',
                        required=False)
    parser.add_argument('--rm_spec_dir',
                        help='Top of beam rm spec directory',
                        required=True)
    parser.add_argument('--bfmt', 
                        help='fstring format of beam name', 
                        default="beam{:05d}_full", 
                        required=False)
    parser.add_argument('--phi_win', type=float,
                        help='Size of RM Spec zoom panel in units of RMSF (def:10 * RMSF)',
                        default=10, required=False)
    parser.add_argument('--imsize', type=float,
                        help='Size of RM sky plot in degrees (def:1.5)',
                        default=1.2, required=False)
    parser.add_argument('--no_pfrac',
                    help='Do NOT make pol frac plot',
                    required=False, action='store_true')
    parser.add_argument('--no_cutouts',
                    help='Do NOT make cutout plots for ALL sources',
                    required=False, action='store_true')
    parser.add_argument('--no_cutouts_select',
                    help='Do NOT make cutout plots for THRESHOLD sources',
                    required=False, action='store_true')
    parser.add_argument('--no_rmplot',
                    help='Do NOT RM image plot',
                    required=False, action='store_true')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Parse command line input
    args = parse_input()

    outdir = args.outdir
    catfile = args.catfile
    fitsfile = args.imfile
    snr_min = args.snr_cut
    outbase = args.outbase
    rm_spec_dir = args.rm_spec_dir
    bformat = args.bfmt
    phi_win = args.phi_win
    imsize = args.imsize

    # Make output directory if needed 
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Filter 
    _, zz = get_filter_rows(catfile, do_filter=True, snr_min=snr_min)

    # Make pfrac plot
    if not args.no_pfrac: 
        print("\nMaking Pol Frac Threshold plot...\n")
        pfrac_outbase = f"{outdir}/{outbase}_pfrac_snr"
        _ = make_pol_frac_plot(catfile, outbase=pfrac_outbase,
                                mark_snr=None, do_filter=True, 
                                mark_nums=zz, title=None)

    # make ALL cutouts
    if not args.no_cutouts:
        print("\nMaking image cutout + RM spec plots...\n")
        all_outdir = f"{outdir}/all"
        if not os.path.exists(all_outdir):
            os.makedirs(all_outdir)
        cut_outbase = f"{all_outdir}/{outbase}_src"
        make_spec_cutouts(catfile, fitsfile, rm_spec_dir, 
                          bstr = bformat, 
                          outbase=cut_outbase, 
                          size_arcsec=120, radius_arcsec=20, frac=0.9,
                          cmap='inferno', phi_win_nrmsf=phi_win)

    # make SELECT cutouts
    if not args.no_cutouts_select:
        print("\nMaking image cutout + RM spec plots for threshold cands...\n")
        top_outdir = f"{outdir}/select"
        if not os.path.exists(top_outdir):
            os.makedirs(top_outdir)
        select_outbase = f"{top_outdir}/{outbase}_select_src"
        make_spec_cutouts(catfile, fitsfile, rm_spec_dir, 
                          bstr = bformat, 
                          outbase=select_outbase, 
                          size_arcsec=120, radius_arcsec=20, frac=0.9,
                          cmap='inferno', use_rows=zz, phi_win_nrmsf=phi_win)

    # make plot of image with rms 
    if not args.no_rmplot:
        print("\nMaking RM sky plot...\n")
        im_out = f"{outdir}/{outbase}_rm_plot"
        make_plot_rms(fitsfile, catfile, use_nums=zz,
                      vrange = None, imsize=imsize, outbase=im_out,
                      clog=False, nocolor='k', dark=False, fwhm=None)
