import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization.wcsaxes import SphericalCircle

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table


def transform_lim(cc0, size_deg, wcs ):
    """
    Get pixel limits so that the image size 
    is size_deg degrees on a side
    """
    ny, nx = wcs.array_shape

    offset = (size_deg/2) * u.deg

    cc_ll = cc0.directional_offset_by(135 * u.deg, offset * np.sqrt(2) )
    cc_ur = cc0.directional_offset_by(-45 * u.deg, offset * np.sqrt(2) )
    
    print(cc_ll)
    print(cc_ur)
    
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

    print(np.min(rms), np.max(rms))
    
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
        vrange = (np.nanmin(dat), np.nanmax(dat))
    print(vrange)
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
 
    ccm = matplotlib.colormaps.get_cmap(col_cmap)
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


