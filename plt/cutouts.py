import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle, Rectangle
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib
from argparse import ArgumentParser
import sys
import os

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def one_beam_to_coord(ra_str, dec_str):
    """
    Convert one ra dec string pair from 
    the beam list to an astropy SkyCoord

    These have format:

    ra_str  = 'hh:mm:ss.ss'
    dec_str = 'dd.mm.ss.ss'

    which is annoying, i know 
    """
    # Careful
    dec_fix = dec_str.replace('.', ':', 2)

    cc = SkyCoord(ra_str, dec_fix, unit=(u.hourangle, u.deg), 
                  frame='fk5')

    return cc


def beamlist_to_coords(bfile):
    """
    Read in a beam list file (npy array)
    and convert to list of SkyCoord objects
    """
    beams = np.load(bfile)
    cc_list = []
    for bb in beams:
        cc = one_beam_to_coord(bb[1], bb[2])
        cc_list.append(cc)

    return cc_list


def get_cutout(cc, data, wcs, size):
    """
    Get the subarray of data centered on the 
    SkyCoord cc using the WCS wcs

    size is the size in pixels
    """
    # Get row and column index for coordinate
    r0, c0 = wcs.world_to_array_index(cc)

    cdat = data[ r0 - size//2 : r0 + size//2, 
                 c0 - size//2 : c0 + size//2 ]

    return cdat, (r0, c0)


def make_one_cutout(cdat, dx_deg, dy_deg, vmin=None, vmax=None,
                    title=None, outfile=None, idx_off=None, xx=None, 
                    radius=None):
    """
    Make a nice cutout image 
    """ 
    if outfile is not None:
        plt.ioff()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cx, cy = cdat.shape
    l_arc = -1 * (cx//2) * dx_deg * 3600
    r_arc = +1 * (cx//2) * dx_deg * 3600
    
    b_arc = -1 * (cy//2) * dy_deg * 3600
    t_arc = +1 * (cy//2) * dy_deg * 3600

    ext = [l_arc, r_arc, b_arc, t_arc] 

    im = ax.imshow(cdat, aspect='equal', origin='lower', 
                   vmin=vmin, vmax=vmax, interpolation='nearest',
                   extent=ext)

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
            ax.text(x, y + 1.25 * radius, f"{xx[ii]:03d}", 
                    color='w', ha='center', va='center', 
                    fontsize=12)
            
    ax.set_xlabel("Offset (arcsec)", fontsize=14)
    ax.set_ylabel("Offset (arcsec)", fontsize=14)
    
    if title is not None:
        ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im)

    if outfile is not None:
        plt.savefig(outfile, dpi=150, bbox_inches=None)
        plt.close()
        plt.ion()

    else:
        plt.show()

    return


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


def find_beams_in_field( idx_arr, idx0, size, rpix):
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


def get_offsets(cc1, cc0, unit='arcmin'):
    """
    calculate the offsets between two coordinates
    """
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

    return (ra_off, dec_off)



def get_cut_stats(cdat, frac=0.9):
    """
    find max at central npix x npix of cdat
    """
    cvals = np.sort(cdat.ravel())
    N = len(cvals)
    xlo = int( N * 0.5 * (1-frac) )
    xhi = int( N * 0.5 * (1+frac) )
    cmids = cvals[ xlo : xhi ]

    sig = np.std(cmids)
    vlo = cmids[0]
    vhi = cmids[-1]

    return sig, vlo, vhi
    


def make_cutouts(cc_list, fits_file, size_arcsec=120, 
                 radius_arcsec=20, frac=0.9, 
                 vmin=None, vmax=None, outdir=None):
    """
    Make cutouts from coord list cc_list
    """
    hdulist = fits.open(fits_file)
    hdu = hdulist[0]

    cdelt1 = hdu.header['CDELT1']
    cdelt2 = hdu.header['CDELT2']

    wcs = WCS(hdu.header)
    
    # Assuming we also have freq and stokes axis
    dat = hdu.data[0, 0, :, :]
    subwcs = wcs[0, 0, :, :]

    # Get center coord
    r0, c0 = subwcs.array_shape
    cc0 = subwcs.array_index_to_world(r0//2 + 1, c0//2 + 1 )

    idx_arr = get_beam_centers(cc_list, subwcs)

    size = int( size_arcsec / (np.abs(cdelt1) * 3600) )
    rpix = int( radius_arcsec / (np.abs(cdelt1) * 3600) )

    for ii, cc in enumerate(cc_list):
        print(ii)
        bname = f"beam{ii:03d}"
        if outdir is not None:
            outname = f"{outdir}/{bname}_cutout.png"
        else:
            outname = f"{bname}_cutout.png"
        ra_off, dec_off = get_offsets(cc, cc0, unit='arcmin')
        cdat, idx0 = get_cutout(cc, dat, subwcs, size)
        if cdat.size == 0:
            continue
        idx_offset, xx = find_beams_in_field( idx_arr, idx0, 
                                              size, rpix)

        title = bname + ": "
        title += "$(\\Delta \\alpha, \\Delta \\delta) =$"
        title += f"({ra_off:+.1f}\', {dec_off:+.1f}\')"

        make_one_cutout(cdat * 1e3, cdelt1, cdelt2, 
                        idx_off=idx_offset, xx=xx, 
                        radius=radius_arcsec, 
                        vmin=vmin, vmax=vmax,
                        title=title, outfile=outname)

    hdulist.close()
    return




def parse_input():
    """
    Parse arguments to cutouts.py
    """
    prog_desc = "Make cutout images for a given set of beams"
    parser = ArgumentParser(description=prog_desc)

    parser.add_argument('--outdir',
                        help='Output directory (def: cwd)',
                        default='.', required=False)
    parser.add_argument('--beamfile',
                        help='npy file of beam coordinates',
                        required=True)
    parser.add_argument('--fitsfile',
                        help='FITS file image',
                        required=True)
    parser.add_argument('--size', type=float,
                        help='Size of cutout image in arcseconds (def=180)',
                        default=180, required=False)
    parser.add_argument('--radius', type=float,
                        help='Region size in arcseconds (def=20)',
                        default=20, required=False)
    parser.add_argument('--frac', type=float,
                        help='Fraction of max to set colorscale (def=0.9)',
                        default=0.9, required=False)
    parser.add_argument('--vmin', type=float,
                        help='Manually set vmin in mJy (default: calc from data)',
                        required=False)
    parser.add_argument('--vmax', type=float,
                        help='Manually set vmax in mJy (default: calc from data)',
                        required=False)

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    # Parse command line input
    args = parse_input()

    outdir = args.outdir
    if not os.path.exists(outdir):
        print(f"Outdir not found: {outdir}")
        sys.exit(0)

    beamfile = args.beamfile
    if not os.path.exists(beamfile):
        print(f"Beam file not found: {beamfile}")
        sys.exit(0)

    fitsfile = args.fitsfile
    if not os.path.exists(fitsfile):
        print(f"FITS file not found: {fitsfile}")
        sys.exit(0)

    cc_list = beamlist_to_coords(beamfile)

    size = args.size
    radius = args.radius
    frac = args.frac
    vmin = args.vmin
    vmax = args.vmax

    make_cutouts(cc_list, fitsfile, size_arcsec=size, 
                 radius_arcsec=radius, frac=frac, 
                 vmin=vmin, vmax=vmax, outdir=outdir)

