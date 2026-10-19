import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import subprocess as sub
import os
import glob
import re
import sys
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import json
from argparse import ArgumentParser
from astropy.table import Table


def blist_to_ccs(blist_file):
    """
    convert beam file to ccs
    """
    ras = []
    decs = []
    blist = np.load(blist_file)
    for ii, col in enumerate(blist):
        ra_str = col[1]
        dec_str = col[2]
        dec_str = ':'.join(dec_str.split('.', 2))
        ras.append(ra_str)
        decs.append(dec_str)

    ccs = SkyCoord(ras, decs, unit=(u.hourangle, u.deg))

    return ccs


def running_median(dd, nwin=30):
    """
    Calculate running median with window size nwin bins 
    """
    mdd = np.copy(dd)
    for ii in range( 0 + nwin//2, len(dd) - nwin//2 ):
        mdd[ii] = np.median( dd[ii - nwin//2 : ii + nwin//2 ] )

    return mdd


def avg_chan(dat, edat, nchan=4):
    """
    average data by nchan chans
    """
    Np = len(dat) // nchan
    dd = np.reshape( dat[: Np * nchan], (-1, nchan) )
    dd_avg = np.mean(dd, axis=1)
    
    edd = np.reshape( edat[: Np * nchan], (-1, nchan) )
    edd_avg = np.sqrt(np.sum( edd**2, axis=1 )) / nchan #/ 10
    
    return dd_avg, edd_avg


def avg_chan_darr(freqs, darr, nchan=4):
    """
    Run channel averaging on darr
    """    
    Np = len(freqs) // nchan
    favg = np.mean( np.reshape(freqs, (-1, nchan)), axis=1 )

    darr_avg = np.zeros( (len(favg), darr.shape[1]), dtype='float')
    for ii in range(4):
        dd_ii, edd_ii = avg_chan(darr[:, 2 * ii], darr[:, 2 *ii+1], nchan=nchan)
        darr_avg[:, 2 * ii] = dd_ii
        darr_avg[:, 2 * ii+1] = edd_ii

    return favg, darr_avg  


def get_bdat(bfile):
    """
    Load in bdat file and return IQUV
    """
    bdat = np.load(bfile)

    I = np.mean(bdat[0, :, :], axis=0)
    Q = np.mean(bdat[1, :, :], axis=0)
    U = np.mean(bdat[2, :, :], axis=0)
    V = np.mean(bdat[3, :, :], axis=0)
    
    #Q -= np.mean(Q)
    #U -= np.mean(U)

    Nt = bdat.shape[1]
    Nf = bdat.shape[2]

    # Uncertainty... this is probably not right
    dI = np.std(bdat[0, :, :], axis=0) / np.sqrt(Nt)
    dQ = np.std(bdat[1, :, :], axis=0) / np.sqrt(Nt)
    dU = np.std(bdat[2, :, :], axis=0) / np.sqrt(Nt)
    dV = np.std(bdat[3, :, :], axis=0) / np.sqrt(Nt)

    dlist = [ I, dI, Q, dQ, U, dU, V, dV ]

    darr = np.vstack( dlist ).T

    return darr


def get_cat_dat(I_row, QU_row):
    """
    Get image catalog data from the Stokes I catalog
    and the QU catalog.  Take as input the relevant
    row from both catalogs.

    For QU, extract the channel fluxes.

    For I, read the fit flux and spectral index, and
    then map that to the QU frequencies

    return darr (like with beam data)
    """
    Q_freqs = QU_row['Q_Freqs']
    U_freqs = QU_row['U_Freqs']

    Q_peaks = QU_row['Q_Aperture_peak']
    U_peaks = QU_row['U_Aperture_peak']

    Q_rms   = QU_row['Q_Aperture_rms']
    U_rms   = QU_row['U_Aperture_rms']

    # Make sure that frequencies are the same,
    # and then drop the pol designation
    if not np.all(Q_freqs == U_freqs):
        print( "Q and U frequencies not the same!")
        print(f" Source_name = {QU_row['Source_name']}")
        print(f" Source_id   = {QU_row['Source_id']}")
        print(f" Gaus_id     = {QU_row['Gaus_id']}")
        return None

    freqs = Q_freqs

    # Check Stokes I spectrum
    Ifit = I_row['Fit_Flux']
    Ialpha = I_row['Alpha']

    if (np.ma.is_masked(Ifit) or np.ma.is_masked(Ialpha) or \
       (Ifit is None) or (Ialpha is None)):
        I  = np.ones(len(freqs))
        dI = np.zeros(len(freqs))
    else:
        I  = Ifit * (freqs / np.mean(freqs))**Ialpha
        dI = np.zeros(len(freqs))

    # For consistency with beamforming, let's make a V
    # column but set it to be all zeros
    V  = np.zeros(len(freqs))
    dV = np.zeros(len(freqs))

    # Set Q and U
    Q  = Q_peaks
    dQ = Q_rms

    U  = U_peaks
    dU = U_rms

    dlist = [ I, dI, Q, dQ, U, dU, V, dV ]
    darr = np.vstack( dlist ).T

    # correct nans
    if np.any(np.isnan(darr)):
        xx = np.where( np.isnan(darr) )
        darr[xx] = 0.0
    
    flag = 0
    if ( np.all( darr[:,2]==0 ) or np.all( darr[:,3] == 0 ) ):
        flag += 1
    if ( np.all( darr[:,4]==0 ) or np.all( darr[:,5] == 0 ) ):
        flag += 1

    return [freqs, darr, flag]



def get_yrange(dat, frac=0.95, pfac=1.2):
    """
    Try to make ylim sensible
    """
    sdat = np.sort(dat)
    N = len(sdat)

    slo = sdat[int(N * (1-frac))]
    shi = sdat[int(N * frac)]

    ylo = slo - pfac * (shi - slo)
    yhi = shi + pfac * (shi - slo)

    return (ylo, yhi)


def stokes_plot(b_darr, b_freqs, im_darr, im_freqs, outfile=None, title=None):
    """
    Compare IQU 
    """
    b_I, b_Q, b_U = b_darr[:, [0,2,4]].T * 1e3
    im_I, im_Q, im_U = im_darr[:, [0,2,4]].T * 1e3

    b_freqs_MHz = b_freqs / 1e6
    im_freqs_MHz = im_freqs / 1e6

    if outfile is not None:
        plt.ioff()

    fig = plt.figure(figsize=(8, 8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    axI = fig.add_subplot(311)
    axQ = fig.add_subplot(312, sharex=axI)
    axU = fig.add_subplot(313, sharex=axI)

    # Plot image value in black, beam in colors
    im_kwargs = {'color' : 'k', 'ls' : '', 'marker' : '.'}
    axI.plot(im_freqs_MHz, im_I, **im_kwargs)
    axI.plot(b_freqs_MHz, b_I, c=colors[0], lw=2)

    axQ.plot(im_freqs_MHz, im_Q, **im_kwargs)
    axQ.plot(b_freqs_MHz, b_Q, c=colors[1], lw=2)
    
    axU.plot(im_freqs_MHz, im_U, **im_kwargs)
    axU.plot(b_freqs_MHz, b_U, c=colors[2], lw=2)

    frac = 0.95
    pfac = 1.2
    I_ylim = get_yrange(b_I, frac=frac, pfac=pfac)
    axI.set_ylim(I_ylim)

    Q_ylim = get_yrange(b_Q, frac=frac, pfac=pfac)
    axQ.set_ylim(Q_ylim)

    U_ylim = get_yrange(b_U, frac=frac, pfac=pfac)
    axU.set_ylim(U_ylim)

    axI.set_ylabel("$S_I \\, \\rm{ (mJy)}$", fontsize=14)
    axQ.set_ylabel("$S_Q \\, \\rm{ (mJy)}$", fontsize=14)
    axU.set_ylabel("$S_U \\, \\rm{ (mJy)}$", fontsize=14)

    tp_kwargs = {'which' : 'major', 'direction': 'in', 'labelbottom' : False,
                 'top': True, 'bottom': True, 'left' : True, 'right' : True,
                 'length' : 5}

    tp_kwargs_bot = tp_kwargs.copy()
    tp_kwargs_bot['labelbottom'] = True

    axI.tick_params(**tp_kwargs)
    axQ.tick_params(**tp_kwargs)
    axU.tick_params(**tp_kwargs_bot)

    g_kwargs = {'alpha' : 0.3 }
    axI.grid(**g_kwargs)
    axQ.grid(**g_kwargs)
    axU.grid(**g_kwargs)

    plt.subplots_adjust(hspace=0.0)

    axU.set_xlabel("Frequency (MHz)", fontsize=14)

    if title is not None:
        axI.set_title(title, fontsize=14)

    if outfile is not None:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
        plt.ion()

    else:
        plt.show()

    return


def many_stokes_plots(npy_list, beam_freqs, I_tab, QU_tab, 
                      outbase, outdir='.', beam_chan_avg=1, 
                      beam_chan_mask=None):
    """
    Make many Stokes comparison plots

    Get the numpy data files for beam data from list of npy files

    Image data is from the catalogs stored in tables I_tab and QU_tab

    Frequencies are given in arrays beam_freqs and im_freqs

    Outbase is the base name of the output plots 
    """
    # Get beam numbers
    bnums = []
    bfiles = []
    for npy_file in npy_list:
        bname = npy_file.split('/')[-1]
        p = re.search("beam([0-9]+)", bname)
        if p is not None:
            bnums.append( int(p.group(1)) )
            bfiles.append( npy_file )
    
    # loop over beams, get data, make plot
    for ii, bnum in enumerate(bnums):
        print(f"beam{bnum:05d}")
        b_darr = get_bdat( bfiles[ii] )
        b_freqs = beam_freqs[:]
        if beam_chan_mask is not None:
            xx = np.where(beam_chan_mask)[0]
            b_darr = b_darr[xx, :]
            b_freqs = b_freqs[xx]

        if beam_chan_avg > 1:
            b_freqs, b_darr = avg_chan_darr(b_freqs, b_darr)

        # Get image data
        im_freqs, im_darr, _ = get_cat_dat(I_tab[bnum], QU_tab[bnum]) 

        # Set output and title
        title = f"beam{bnum:05d}"
        outfile = f"{outdir}/{outbase}_beam{bnum:05d}.png"

        stokes_plot(b_darr, b_freqs, im_darr, im_freqs, 
                    outfile=outfile, title=title)

    return



#################################################



def read_FDF(datfile):
    """
    Read in faraday depth spectra

    Data in form (phi_rad_m2, real spec, imag spec)
    """
    dat = np.loadtxt(datfile)
    phis = dat[:, 0]
    r_spec = dat[:, 1]
    i_spec = dat[:, 2]
    return phis, r_spec, i_spec


def make_clean_plot(datfile, rm=None, show_ri=False, xlim=None,
                    title=None, outfile=None):
    """
    Make a nice plot of clean FDF
    
    mark known rm at `rm` if not none
    """
    if outfile is not None:
        plt.ioff()

    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)

    phis, r_spec, i_spec = read_FDF(datfile)

    # Put in uJy
    r_spec *= 1e6
    i_spec *= 1e6

    m_spec = np.sqrt(r_spec**2 + i_spec**2)
    
    if show_ri:
        ax.plot(phis, r_spec, lw=1, alpha=0.5, label="Real")
        ax.plot(phis, i_spec, lw=1, alpha=0.5, label="Imag")
    
    ax.plot(phis, m_spec, lw=1.5, c='k', label="Mag")

    if rm is not None:
        ax.axvline(x=rm, ls='--', c='r', alpha=0.5, zorder=0)

    ax.set_xlabel("$\\phi~({\\rm rad~m}^{-2})$", fontsize=16)
    ax.set_ylabel("${\\rm Flux~Density}~(\\mu{\\rm Jy~beam}^{-1})$", 
                  fontsize=14)

    max_val = np.max(m_spec)
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

    if title is not None:
        plt.title(title, fontsize=16)

    if outfile is not None:
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        plt.close()
        plt.ion()
    else:
        plt.show()

    return


def make_psr_rm_plots(bnums, bdir, nbeams, 
                      psr_names, psr_nums, psr_rms):
    """
    Make plots for pulsars
    """
    for bnum in bnums:
        bpsr = (bnum // nbeams) * nbeams
        print(bpsr)
        xx = np.where( psr_nums == bpsr)[0]
        print(xx)
        if len(xx) == 0:
            continue
        pname = psr_names[xx[0]]
        prm = psr_rms[xx[0]]
    
        #bname = f"beam{bnum:03d}_full_split"
        bname = f"beam{bnum:03d}_full_split"
        datfile = f"{bdir}/{bname}/{bname}_FDFclean.dat"
        print(datfile)
   
        outfile = f"{pname}_beam{bnum:03d}.png"
        xlim = (-7e4, 7e4)
        make_clean_plot(datfile, rm=prm, show_ri=False, xlim=xlim,
                        title=pname, outfile=outfile)
        
    return


def make_rm_plots(bnums, bdir, xlim=(-1e4, 1e4)):
    """
    Make rm plots
    """
    for bnum in bnums:
        bname = f"beam{bnum:03d}_full"
        datfile = f"{bdir}/{bname}/{bname}_FDFclean.dat"
        print(datfile)
   
        outfile = f"beam{bnum:03d}_rm_clean.png"
        make_clean_plot(datfile, rm=None, show_ri=False, xlim=xlim,
                        title=None, outfile=outfile)
        
    return


def json_to_cat(bnum, json_file):
    """
    read in json file results for clean, and convert 
    to a string to be print in catalog
    """
    # Need to check if json file exists. For image 
    # RM measurement, some may not exist, so we'll just 
    # put nans
    if os.path.exists(json_file): 
        with open(json_file, "r") as fin:
            dd = json.load(fin)

        rm = dd['phiPeakPIfit_rm2']
        rm_err = dd['dPhiPeakPIfit_rm2']

        # Convert to mJy
        PI = dd['ampPeakPIfit'] * 1e3
        PI_err = dd['dAmpPeakPIfit'] * 1e3

        PI_snr = dd['snrPIfit']

        PA0 = dd['polAngle0Fit_deg']
        PA0_err = dd['dPolAngle0Fit_deg']
    else:
        rm = np.nan
        rm_err = np.nan
        PI = np.nan
        PI_err = np.nan
        PI_snr = np.nan
        PI_snr = np.nan
        PA0 = np.nan
        PA0_err = np.nan

    ostr = f"{bnum:03d}   {rm:10.2f}  {rm_err:10.2f}  " +\
           f"{PI:8.3f}  {PI_err:8.3f}  {PI_snr:7.1f}  "+\
           f"{PA0:8.2f}  {PA0_err:8.2f} \n"

    return ostr
    

def json_to_cat_coord(bnum, json_file, ra, dec):
    """
    read in json file results for clean, and convert 
    to a string to be print in catalog
    """
    # Need to check if json file exists. For image 
    # RM measurement, some may not exist, so we'll just 
    # put nans
    if os.path.exists(json_file): 
        with open(json_file, "r") as fin:
            dd = json.load(fin)

        rm = dd['phiPeakPIfit_rm2']
        rm_err = dd['dPhiPeakPIfit_rm2']

        # Convert to mJy
        PI = dd['ampPeakPIfit'] * 1e3
        PI_err = dd['dAmpPeakPIfit'] * 1e3

        PI_snr = dd['snrPIfit']

        PA0 = dd['polAngle0Fit_deg']
        PA0_err = dd['dPolAngle0Fit_deg']
    else:
        rm = np.nan
        rm_err = np.nan
        PI = np.nan
        PI_err = np.nan
        PI_snr = np.nan
        PI_snr = np.nan
        PA0 = np.nan
        PA0_err = np.nan

    ostr = f"{bnum:03d}   {rm:10.2f}  {rm_err:10.2f}  " +\
           f"{PI:8.3f}  {PI_err:8.3f}  {PI_snr:7.1f}  "+\
           f"{PA0:8.2f}  {PA0_err:8.2f}  {ra:10.5f}  {dec:10.5f}\n"

    return ostr
    

def make_catalog(bdir, outfile):
    """
    Parse all the RMclean json files for every beam
    and write values of interest to the catalog
    """
    blist = glob.glob(f"{bdir}/*beam[0-9]*")
    blist.sort()

    bnames = []
    bnums = []
    
    for bb in blist:
        bname = bb.split('/')[-1]
        p = re.search("beam([0-9]+)", bname)
        if p is not None:
            bnames.append(bname)
            bnums.append( int(p.group(1)) )

    # get cat lines
    olines = []
    for ii, bdir in enumerate(blist):
        jfile = f"{bdir}/{bnames[ii]}_RMclean.json"
        ostr = json_to_cat(bnums[ii], jfile)
        olines.append(ostr)

    # Write catalog 
    with open(outfile, "w") as fout:
        hdr1 = f"#{'Beam':^5} {'phi':^10}  {'phi_err':^10}  "+\
               f"{'PI':^8}  {'PI_err':^8}  {'SNR':^7}  "+\
               f"{'PA0':^8}  {'PA0_err':^8} \n"
        hdr2 = f"#{'':^5} {'(rad/m^2)':^10}  {'(rad/m^2)':^10}  "+\
               f"{'(mJy)':^8}  {'(mJy)':^8}  {'':^7}  "+\
               f"{'(deg)':^8}  {'(deg)':^8} \n"
        hdr3 = "#" + "="*len(hdr1) + "\n"
        fout.write(hdr1)
        fout.write(hdr2)
        fout.write(hdr3)
        for oline in olines:
            fout.write(oline)
    return


def make_catalog2(srcfile, bdir, outfile):
    """
    Parse all the RMclean json files for every beam
    and write values of interest to the catalog

    Add ra and dec from fits cat
    """
    blist = glob.glob(f"{bdir}/beam*")
    #blist.sort()

    bnames = [bb.split('/')[-1] for bb in blist]
    bnums  = [int( ff.lstrip('beam').split('_')[0] ) for ff in bnames]

    xx = np.argsort(bnums)
    
    bnums = [bnums[ii] for ii in xx ]
    bnames = [bnames[ii] for ii in xx ]
    blist = [blist[ii] for ii in xx ]

    tab = Table.read(srcfile)
    ras  = tab['RA'].value
    decs = tab['DEC'].value

    print(type(ras[0]))
    
    # get cat lines
    olines = []
    for ii, bdir in enumerate(blist):
        jfile = f"{bdir}/{bnames[ii]}_RMclean.json"
        bb = bnums[ii]
        ostr = json_to_cat_coord(bb, jfile, ras[bb], decs[bb])
        olines.append(ostr)

    # Write catalog 
    with open(outfile, "w") as fout:
        hdr1 = f"#{'Beam':^5} {'phi':^10}  {'phi_err':^10}  "+\
               f"{'PI':^8}  {'PI_err':^8}  {'SNR':^7}  "+\
               f"{'PA0':^8}  {'PA0_err':^8} {'RA_deg':^10}  {'DEC_deg':^10}\n"
        hdr2 = f"#{'':^5} {'(rad/m^2)':^10}  {'(rad/m^2)':^10}  "+\
               f"{'(mJy)':^8}  {'(mJy)':^8}  {'':^7}  "+\
               f"{'(deg)':^8}  {'(deg)':^8}  {'(deg)':^10}  {'(deg)':^10}\n"
        hdr3 = "#" + "="*len(hdr1) + "\n"
        fout.write(hdr1)
        fout.write(hdr2)
        fout.write(hdr3)
        for oline in olines:
            fout.write(oline)
    return


def get_rm_idx(rm_txt, snr_min=-1):
    """
    Read rm cat and get rows meeting snr min criterion 
    """
    rdat = np.loadtxt(rm_txt)
    bnums = rdat[:, 0].astype('int')
    rms = rdat[:, 1]
    pflux = rdat[:, 3]
    snrs = rdat[:, 5]
    xx = np.where( snrs > snr_min )[0]
    return bnums[xx]
    


def plot_rms(infile, rm_txt, cc0, rm0=0, vmin=None, vmax=None, 
             cattype='blist', snr_min=-1, frame='fk5', offset=True):
    """
    Make a plot using coords from cat_fits and 
    rm values from rm_txt centered on cc0
    """
    if cattype == 'blist':
        cc = blist_to_ccs(infile)
    else:
        tab = Table.read(infile)
        cc = SkyCoord(tab["RA"], tab["DEC"])

    rdat = np.loadtxt(rm_txt)
    bnums = rdat[:, 0].astype('int')
    rms = rdat[:, 1]
    pflux = rdat[:, 3]
    snrs = rdat[:, 5]
    xx = np.where( snrs > snr_min )[0]

    if offset:
        if frame == 'fk5':
            dra = (cc.fk5.ra - cc0.fk5.ra) * np.cos(cc0.fk5.dec)
            ddec = (cc.fk5.dec - cc0.fk5.dec)

            dx = dra.arcmin
            dy = ddec.arcmin

        else:
            dlon = (cc.galactic.l - cc0.galactic.l) * np.cos(cc0.galactic.b)
            dlat = (cc.galactic.b - cc0.galactic.b)
        
            dx = dlon.arcmin
            dy = dlat.arcmin

    else:
        if frame == 'fk5':
            dx = cc.fk5.ra
            dy = cc.fk5.dec

        else:
            dx = cc.galactic.l.deg
            dy = cc.galactic.b.deg

            #yy = np.where( dx > 180 )[0]
            #print(yy)
            #if len(yy):
            #    dx[yy] = dx[yy] - 360 
            #    print(dx)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_aspect('equal')
    #ax.plot(dx, dy, marker='o', ls='')

    print(np.mean(rms))
    cax = ax.scatter(dx[xx], dy[xx], marker='o', c=rms[xx]-rm0, cmap='coolwarm', 
                     s=3 * (snrs[xx]), edgecolor='k', lw=0.5,
                     vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cax, shrink=0.95, extend='both')
    cbar.ax.set_ylabel('${\\rm RM}~({\\rm rad~m}^{-2})$', fontsize=12)

    xlim = ax.get_xlim()
    ax.set_xlim( xlim[1], xlim[0])
   
    if frame == 'fk5': 
        if offset:
            ax.set_xlabel("RA Offset (arcmin)", fontsize=14)
            ax.set_ylabel("DEC Offset (arcmin)", fontsize=14)
        else:
            ax.set_xlabel("RA (deg)", fontsize=14)
            ax.set_ylabel("DEC (deg)", fontsize=14)
    else:
        if offset:
            ax.set_xlabel("Gal Lon Offset (arcmin)", fontsize=14)
            ax.set_ylabel("Gal Lat Offset (arcmin)", fontsize=14)
        else:
            ax.set_xlabel("Gal Lon (deg)", fontsize=14)
            ax.set_ylabel("Gal Lat (deg)", fontsize=14)

    #ax.plot(0, 0, marker='x', c='r')

    plt.show()
    return
    


def plot_frac_pol(infile, rm_txt, cc0, vmin=None, vmax=None, 
                  snr_min=-1, frame='fk5'):
    """
    Make a plot using coords from cat_fits and 
    rm values from rm_txt centered on cc0
    """
    tab = Table.read(infile)
    cc = SkyCoord(tab["RA"], tab["DEC"])
    fluxes = tab['Total_flux'] * 1e3

    rdat = np.loadtxt(rm_txt)
    bnums = rdat[:, 0].astype('int')
    rms = rdat[:, 1]
    pflux = rdat[:, 3]
    snrs = rdat[:, 5]
    pfrac = pflux / fluxes
    xx = np.where( snrs > snr_min )[0]

    if frame == 'fk5':
        dx = cc.fk5.ra.deg
        dy = cc.fk5.dec.deg

    else:
        dx = cc.galactic.l.deg
        dy = cc.galactic.b.deg

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_aspect('equal')
    #ax.plot(dx, dy, marker='o', ls='')

    cax = ax.scatter(dx[xx], dy[xx], marker='o', c=pfrac[xx], cmap='inferno', 
                     s=3 * (snrs[xx]), edgecolor='k', lw=0.5,  
                     norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(cax, shrink=0.95)
    cbar.ax.set_ylabel('Lin Pol Frac', fontsize=12)

    xlim = ax.get_xlim()
    ax.set_xlim( xlim[1], xlim[0])
   
    if frame == 'fk5': 
       ax.set_xlabel("RA (deg)", fontsize=14)
       ax.set_ylabel("DEC (deg)", fontsize=14)
    else:
       ax.set_xlabel("Gal Lon (deg)", fontsize=14)
       ax.set_ylabel("Gal Lat (deg)", fontsize=14)

    #ax.plot(0, 0, marker='x', c='r')
    plt.show()
    return


def parse_input():
    """
    Parse arguments to Image cat RM synthesis
    """
    prog_desc = "Run RM synthsis on QU spectra from catalog"
    parser = ArgumentParser(description=prog_desc)

    parser.add_argument('--outdir',
                        help='Output directory (def: cwd)',
                        default='.', required=False)
    parser.add_argument('--outbase',
                        help='Basename for output plots',
                        default='compare', required=False)
    parser.add_argument('--Icat',
                        help='Stokes I source catalog',
                        required=True)
    parser.add_argument('--QUcat',
                        help='Stokes QU source catalog',
                        required=True)
    parser.add_argument('--freqfile',
                        help='npy array of channel frequencies',
                        required=True)
    parser.add_argument('--maskfile',
                        help='npy array of channel mask for beam data',
                        required=False)
    parser.add_argument('--beam_chan_avg', type=int, 
                        help='number of channels to average in beam data',
                        default=1, required=False)
    parser.add_argument('dat_files', nargs='+',
                        help='Beam data npy file(s) to plot')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Parse command line input
    args = parse_input()

    outbase = args.outbase
    
    # Get freq file
    freqfile = args.freqfile
    if not os.path.exists(freqfile):
        print(f"freqfile not found: {freqfile}")
        sys.exit(0)
    beam_freqs = np.load(freqfile)

    # Get mask file (if nec)
    maskfile = args.maskfile
    if maskfile is not None:
        if not os.path.exists(maskfile):
            print(f"Mask file not found: {maskfile}")
            sys.exit(0)
        mask = np.load(maskfile)
    else:
        mask = None

    # get output directory
    outdir = args.outdir
    if not os.path.exists(outdir):
        print(f"Outdir not found: {outdir}")
        sys.exit(0)

    # Open cats and make sure they are the same size
    I_tab  = Table.read(args.Icat)
    QU_tab = Table.read(args.QUcat)

    if (len(I_tab) != len(QU_tab)):
        print("I and QU catalogs have different number of rows!")
        print(f"len(Icat)  = {len(I_tab)}")
        print(f"len(QUcat) = {len(QU_tab)}")
        sys.exit(0)

    # Get beam chan avg
    chan_avg = args.beam_chan_avg

    # get numpy data
    npy_list = args.dat_files

    many_stokes_plots(npy_list, beam_freqs, I_tab, QU_tab, 
                      outbase, outdir=outdir, 
                      beam_chan_avg=chan_avg, 
                      beam_chan_mask=mask)
