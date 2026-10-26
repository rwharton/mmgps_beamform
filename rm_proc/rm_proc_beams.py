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
import multiprocessing
from contextlib import closing

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
        print(f" Source_name = {QU['Source_name']}")
        print(f" Source_id   = {QU['Source_id']}")
        print(f" Gaus_id     = {QU['Gaus_id']}")
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

    return [freqs, darr]



def polarr_to_txt(outfile, freqs, Qarr, Uarr, dQ, dU):
    """
    write text file appropriate for rmsynth1d
    """
    with open(outfile, 'w') as fout:
        for ii, freq in enumerate(freqs):
            ostr = f"{freq:12.1f} " +\
                   f"{Qarr[ii]:10.6f} {Uarr[ii]:10.6f} " +\
                   f"{dQ:10.6f} {dU:10.6f} \n"

            fout.write(ostr)

    return



def darr_to_txt(outfile, darr, freqs, skip_zeros=False):
    """
    Write spectra to text file format that 
    can be passed to RMSynth1D

    'RMsynth1D expects the input to be the 
     form of an ASCII text file, with 7 
     columns: frequency (in Hz), Stokes I, 
     Stokes Q, Stokes U, error in I, error in Q, 
     error in U. Each row is a single channel. 
     No header is expected.'
    
    Can also provide 5 columns and skip I

    Can also mask with NaNs
    """
    with open(outfile, 'w') as fout:
        for ii, freq in enumerate(freqs):
            xx = [0, 2, 4, 1, 3, 5]
            I, Q, U, dI, dQ, dU = darr[ii, xx]

            if (skip_zeros==True) and (I==0):
                continue
            
            ostr = f"{freq:12.1f} " +\
                   f"{I:10.6f} {Q:10.6f} {U:10.6f} " +\
                   f"{dI:10.6f} {dQ:10.6f} {dU:10.6f} \n"

            fout.write(ostr)

    return


def run_rmsynth1d(infile, phimax=None):
    """
    Run rmsynth1d on the input text file infile 
    (assumed to be in proper format).  Optionally 
    include a phimax, otherwise the default will 
    be used (which depends on particular frequency 
    channel arrangement)
    """
    phi_str = ""
    if phimax is not None:
        phi_str = f"-l {phimax:.2f}"
    
    cmd = f"rmsynth1d -i -S {phi_str} {infile}"
    #cmd = f"rmsynth1d -S {phi_str} {infile}"
    print(cmd)
    
    try:
        ret = sub.run(cmd, shell=True, check=True)
    except sub.CalledProcessError:
        print(f"cmd failed: {cmd}")
    except:
        print("Something else failed somehow")

    return ret.returncode 


def run_rmclean1d(infile, niter, threshold):
    """
    Run rmclean1d on the data set given 
    by the input text file.  Assumes that 
    rmsynth1d has already been run.  

    niter = number of clean interations

    threshold = cleaning threshold in Jy (if positive)
                or in sigma (if negative) 
    """
    cmd = f"rmclean1d -S -n {niter} -c {threshold} {infile}"
    print(cmd)
    
    try:
        ret = sub.run(cmd, shell=True, check=True)
    except sub.CalledProcessError:
        print(f"cmd failed: {cmd}")
    except:
        print("Something else failed somehow")

    return ret.returncode 


def one_rm_clean(npy_file, freq_file, mask_file=None, outdir='.', 
                 niter=200, threshold=-2, phimax=None, skip_zeros=False):
    """
    Run full RM processing on one npy beam file
    """
    fname = npy_file.split('/')[-1]
    basenm = fname.split('.npy')[0]
    
    # directory where clean data will go
    bdir = f"{outdir}/{basenm}"
    if not os.path.exists(bdir):
        os.mkdir(bdir)
    else:
        print(f"{bdir} already exists!  skipping")
        return

    # read in npy data and write to text 
    # file for rm synthesis and clean
    freqs = np.load(freq_file)
    darr = get_bdat(npy_file)   
    #print(darr.shape)
    if mask_file is not None:
        mask = np.load(mask_file)
        xx = np.where( mask )[0]
    else:
        xx = np.arange(len(freqs))
    dat_file = f"{bdir}/{basenm}.txt"
    darr_to_txt(dat_file, darr[xx], freqs[xx], skip_zeros=skip_zeros)

    # run rm synthesis
    ret1 = run_rmsynth1d(dat_file, phimax=phimax)
    
    # run rm clean
    ret2 = run_rmclean1d(dat_file, niter, threshold)

    return


def multi_rm_clean(nproc, npy_files, freq_file, mask_file=None, 
                   outdir='.', niter=200, threshold=-2, 
                   phimax=None, skip_zeros=False):
    """
    Given a list of npy data files, run rm clean on 
    all with multiprocessing using nproc cores, 
     and output to individual directories
    """
    with closing(multiprocessing.Pool(processes=nproc)) as pool:
        results = [pool.apply_async(one_rm_clean,
                   args=(npy_file, freq_file, mask_file, outdir, 
                         niter, threshold, phimax, skip_zeros) )\
                         for npy_file in npy_files]
        all_beams = [p.get() for p in results]
    return


def rm_clean_many(npy_files, freq_file, mask_file=None, outdir='.', 
                  niter=200, threshold=-2, phimax=None):
    """
    Given a list of npy data files, run rm clean on 
    all and output to individual directories
    """
    for npy_file in npy_files:
        fname = npy_file.split('/')[-1]
        basenm = fname.split('.npy')[0]
        
        # directory where clean data will go
        bdir = f"{outdir}/{basenm}"
        if not os.path.exists(bdir):
            os.mkdir(bdir)
        else:
            print(f"{bdir} already exists!  skipping")
            continue

        # read in npy data and write to text 
        # file for rm synthesis and clean
        freqs = np.load(freq_file)
        darr = get_bdat(npy_file)   
        if mask_file is not None:
            mask = np.load(mask_file)
            xx = np.where( mask )[0]
        else:
            xx = np.arange(len(freqs))
        dat_file = f"{bdir}/{basenm}.txt"
        darr_to_txt(dat_file, darr[xx], freqs[xx])

        # run rm synthesis
        ret1 = run_rmsynth1d(dat_file, phimax=phimax)
        
        # run rm clean
        ret2 = run_rmclean1d(dat_file, niter, threshold)
    
    return


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


def json_to_parr(bnum, json_file, use_obs=False):
    """
    read in json file results for clean, and return an 
    array of the relevant parameters:

    RM, E_RM, PI, E_PI, PA0, E_PA0, PI_snr

    Fluxes are in Jy
    """
    with open(json_file, "r") as fin:
        dd = json.load(fin)

    # Get Values
    rm = dd['phiPeakPIfit_rm2']
    PI = dd['ampPeakPIfit'] 
    PA0 = dd['polAngle0Fit_deg']
   
    # Get errs 
    if use_obs: 
        rm_err = dd['dPhiObserved_rm2']
        PI_err = dd['dAmpObserved']
        PA0_err = dd['dPolAngleFit0Observed_deg']
    else:
        rm_err = dd['dPhiPeakPIfit_rm2']
        PI_err = dd['dAmpPeakPIfit'] 
        PA0_err = dd['dPolAngle0Fit_deg']
    
    PI_snr = PI / PI_err

    parr = np.array([ rm, rm_err, PI, PI_err, PA0, PA0_err, PI_snr ])

    return parr


def json_to_cat(bnum, json_file, use_obs=False):
    """
    read in json file results for clean, and convert 
    to a string to be print in catalog
    """
    with open(json_file, "r") as fin:
        dd = json.load(fin)

    # Get Values
    rm = dd['phiPeakPIfit_rm2']
    PI = dd['ampPeakPIfit'] * 1e3  # mJy
    PA0 = dd['polAngle0Fit_deg']
   
    # Get errs 
    if use_obs: 
        rm_err = dd['dPhiObserved_rm2']
        PI_err = dd['dAmpObserved'] * 1e3
        PA0_err = dd['dPolAngleFit0Observed_deg']
    else:
        rm_err = dd['dPhiPeakPIfit_rm2']
        PI_err = dd['dAmpPeakPIfit'] * 1e3
        PA0_err = dd['dPolAngle0Fit_deg']
    
    #PI_snr = dd['snrPIfit']
    PI_snr = PI / PI_err

    ostr = f"{bnum:03d}   {rm:10.2f}  {rm_err:10.2f}  " +\
           f"{PI:8.3f}  {PI_err:8.3f}  {PI_snr:7.1f}  "+\
           f"{PA0:8.2f}  {PA0_err:8.2f} \n"

    return ostr
    

def json_to_cat_coord(bnum, json_file, ra, dec):
    """
    read in json file results for clean, and convert 
    to a string to be print in catalog
    """
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

    ostr = f"{bnum:03d}   {rm:10.2f}  {rm_err:10.2f}  " +\
           f"{PI:8.3f}  {PI_err:8.3f}  {PI_snr:7.1f}  "+\
           f"{PA0:8.2f}  {PA0_err:8.2f}  {ra:10.5f}  {dec:10.5f}\n"

    return ostr
    

def make_catalog(bdir, outfile, use_obs=False):
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
        ostr = json_to_cat(bnums[ii], jfile, use_obs=use_obs)
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



def make_fits_catalog(catfile, bdir, use_obs=False):
    """
    Parse all the RMclean json files for every beam
    and make a copy of the catfile, appending the 
    parameters of interest to the source catalog FITS file
    """
    # Open catalog FITS file
    tab = Table.read(catfile)
    N = len(tab)

    # RM units
    uRM = u.rad / u.m**2 

    # Prepare to add columns
    col_names = ['RM', 'E_RM', 'PI', 'E_PI', 'PA0', 'E_PA0', 'PI_SNR']
    col_units = [uRM ,  uRM  , u.Jy,  u.Jy,  u.deg, u.deg, 1 ]

    nan_arr = np.array([np.nan] * len(tab))

    # If column doesn't exist, add it with default values
    # If it does exist, reset to default values 
    for ii in range(len(col_names)):
        cname = col_names[ii]
        cunit = col_units[ii]
        nval = np.nan * cunit

        if cname in tab.colnames:
            tab[cname] = nval
        else:
            tab.add_column(nval, name=cname)
    
    # Go through beams directory and get beam names and numbers
    # The beam numbers correspond to the table row
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
    for ii, beam_dir in enumerate(blist):
        jfile = f"{beam_dir}/{bnames[ii]}_RMclean.json"
        bnum = bnums[ii]
        parr = json_to_parr(bnum, jfile, use_obs=use_obs)

        for jj, ck in enumerate(col_names):
            pval = parr[jj] #* col_units[jj]
            tab[bnum][ck] = pval

    cat_fname = os.path.basename(catfile)
    out_base = cat_fname.rsplit('.fits', 1)[0]
    outfile = f"{bdir}/{out_base}_RM.fits" 
    tab.write(outfile)

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




def parse_input():
    """
    Parse arguments to MS beamform
    """
    prog_desc = "Make time and spectrum plots for IQUV data in beams"
    parser = ArgumentParser(description=prog_desc)

    parser.add_argument('--outdir',
                        help='Output directory (def: cwd)',
                        default='.', required=False)
    parser.add_argument('--freqfile',
                        help='npy array of channel frequencies in MHz',
                        required=True)
    parser.add_argument('--maskfile',
                        help='npy array of channel mask (1=use chan, 0=skip chan)',
                        required=False)
    parser.add_argument('--keep_zeros', help='Do NOT flag zeros in spectrum (default = False)', 
                        action='store_true')
    parser.add_argument('--niter', type=int,
                        help='Number of CLEAN iterations (def=200)',
                        default=200, required=False)
    parser.add_argument('--threshold', type=float,
                        help='CLEAN threshold in (neg) sigma (def=-2)',
                        default=-2, required=False)
    parser.add_argument('--phimax', type=float,
                        help='Maximum Faraday depth (rad/^2, def: -1, no max)',
                        default=-1, required=False)
    parser.add_argument('--catfile',
                        help='FITS catalog file to add results to',
                        required=True)
    parser.add_argument('--use_obs_errs', help='Use obs errors from RM-tools in catalog (default = False)', 
                        action='store_true')
    parser.add_argument('--nproc', type=int,
                        help='Number of parallel processes to run (def=1)',
                        default=1, required=False)
    parser.add_argument('dat_files', nargs='+',
                        help='Beam data file(s) for RM CLEAN-ing')

    args = parser.parse_args()

    return args


    
if __name__ == "__main__":
    # Parse command line input
    args = parse_input()

    freqfile = args.freqfile
    if not os.path.exists(freqfile):
        print(f"Freq file not found: {freqfile}")
        sys.exit(0)

    maskfile = args.maskfile
    if maskfile is not None:
        if not os.path.exists(maskfile):
            print(f"Mask file not found: {maskfile}")
            sys.exit(0)

    # Make output directory if needed
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    phimax = args.phimax
    if phimax <= 0:
        phimax = None

    if args.keep_zeros:
        mask_zeros = False
    else:
        mask_zeros = True
        
    niter = args.niter
    threshold = args.threshold
    dat_files = args.dat_files
    
    nproc = args.nproc
    multi_rm_clean(nproc, dat_files, freqfile, mask_file=maskfile, 
                   outdir=outdir, niter=niter, threshold=threshold, 
                   phimax=phimax, skip_zeros=mask_zeros)

    catfile = args.catfile
    make_fits_catalog(catfile, outdir, use_obs=args.use_obs_errs)
