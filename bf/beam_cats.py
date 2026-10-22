import numpy as np
from collections import Counter

from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

def make_beam_list(coords):
    """
    Take list of astropy SkyCoord objects and
    convert them to a string format that will
    be used by the beamforming pipeline
    """
    blist = []
    for cc in coords:
        ra_str  = cc.ra.to_string(u.hour, sep=':', pad=True)
        dec_str = cc.dec.to_string(u.deg, sep='.', pad=True,
                                   alwayssign=True)
        blist.append( ["J2000", ra_str, dec_str] )
    blist = np.array(blist, dtype=str)
    return blist


def table_to_blist(full_tab, userows=[]):
    """
    Get coords of sources in Table (optionally selecting
    a list of rows) and return a string format list of coords
    to be used for beamlist
    """
    if len(userows):
        tab = full_tab[userows]
    else:
        tab = full_tab

    cc_list = []

    u_ra  = tab['RA'].unit
    u_dec = tab['DEC'].unit

    for trow in tab:
        ra  = trow['RA']
        dec = trow['DEC']

        cc = SkyCoord(ra, dec, unit=(u_ra, u_dec))
        cc_list.append(cc)

    blist = make_beam_list(cc_list)

    return blist


def cat_to_blist(fits_file, userows=[]):
    """
    Get coords of sources in fits_file (optionally selecting
    a list of rows) and return a string format list of coords
    to be used for beamlist
    """
    full_tab = Table.read(fits_file)
    blist = table_to_blist(full_tab, userows=userows)

    return blist


def table_to_coords(fits_file, userows=[]):
    """
    Get coords of sources in fits_file 
    (optionally selecting a list of rows)
    """
    full_tab = Table.read(fits_file)

    if len(userows):
        tab = full_tab[userows]
    else:
        tab = full_tab

    cc_list = []

    u_ra  = tab['RA'].unit
    u_dec = tab['DEC'].unit

    for trow in tab:
        ra  = trow['RA']
        dec = trow['DEC']

        cc = SkyCoord(ra, dec, unit=(u_ra, u_dec))
        cc_list.append(cc)

    return cc_list



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


def filter_nearby(fits_file, min_sep=0.0, min_flux=0,
                  userows=[]):
    """
    sort by flux and filter out sources that are closer
    than min_sep from a brighter source

    can optionally can only do this down to min_flux
    """
    full_tab = Table.read(fits_file)

    if len(userows):
        tab = full_tab[userows]
    else:
        tab = full_tab

    flux = tab['Total_flux']

    u_ra  = tab['RA'].unit
    u_dec = tab['DEC'].unit

    ras  = tab['RA']
    decs = tab['DEC']

    coords = SkyCoord(ras, decs, unit=(u_ra, u_dec))

    idx_bad = []
    xx = np.argsort(flux)[::-1]

    flux_srt = flux[xx]
    coord_srt = coords[xx]

    for ii, fsrt in enumerate(flux_srt):
        if fsrt <= min_flux:
            break

        if ii == len(flux_srt) - 1:
            break

        if ii in idx_bad:
            continue

        cc_ii = coord_srt[ii]
        cc_rest = coord_srt[ii+1:]

        seps = cc_ii.separation(cc_rest).arcsec

        yy = np.where(seps <=  min_sep)[0]

        if len(yy):
            cur_bad = (yy+ii+1).tolist()
            idx_bad += cur_bad
    idx_bad_arr = np.unique( np.array( idx_bad ) )
    idx_good = np.setdiff1d( np.arange(len(xx)), idx_bad_arr )

    if len(userows):
        xx_out = userows[xx[idx_good]]
    else:
        xx_out = xx[idx_good]

    return xx_out


def get_flux_sorted_rows(fits_file, smin=0, point=True):
    """
    Get flux sorted rows of fits file to give to blist

    Optionally give a cutoff
    """
    tab = Table.read(fits_file)

    flux = tab['Total_flux']

    xx = np.argsort( flux )[::-1]
    yy = np.where( flux[xx] >= smin )[0]
    idx = xx[yy]

    return idx


def loc_filter(fits_file, cc0, sep_arcmin):
    """
    return rows of sources that fall within sep_arcmin
    arcminutes from the sky coordinate cc0
    """
    tab = Table.read(fits_file)
    ccs = SkyCoord(tab['RA'], tab['DEC'])
    
    if sep_arcmin <= 0:
        xx = np.arange(len(tab), dtype='int')
    else:
        xx = np.where( cc0.separation(ccs).arcmin < sep_arcmin )[0]
    
    return xx


def tab_PC_filter(tab, sep_arcmin):
    """
    return table of sources that fall within sep_arcmin
    arcminutes from the phase center as determined by 
    the table column 'Sep_PC'
    """
    seps = tab['Sep_PC'].value * 60

    if sep_arcmin <= 0:
        xx = np.arange(len(tab), dtype='int')
    else:
        xx = np.where( seps < sep_arcmin )[0]

    tab_out = tab[xx]

    return tab_out


def tab_isl_filter(full_tab, max_comp=1, min_flux=0, userows=[]):
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

    tab_out = tab[xx_out]

    return tab_out


def tab_prefilter(tab, sep_arcmin=-1.0, filter_isl=False):
    """
    Do preliminary table filtering on catalog table

    If sep_arcmin <= 0, it is skipped

    If filter_isl is False, it is skipped

    Return table with selected rows
    """
    # First get table rows for sources with sep_arcmin arcminutes 
    # of the phase center if desired
    if sep_arcmin > 0:
        tab = tab_PC_filter(tab, sep_arcmin)
    
    # Now filter by island if desired
    if filter_isl:
        tab = tab_isl_filter(tab)
    
    return tab
