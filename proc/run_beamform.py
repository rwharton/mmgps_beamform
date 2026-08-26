import numpy as np
import multiprocessing
import time, sys, glob, os, re
from functools import partial
from contextlib import closing
import gc
import casatools
from argparse import ArgumentParser
from datetime import datetime

import beam_extract as be
import beam_combine as bc

ms = casatools.ms()
msmd = casatools.msmetadata()
qa = casatools.quanta()
me = casatools.measures()


def parse_input():
    """
    Parse arguments to MS beamform
    """
    prog_desc = "Extract beamformed IQUV dynamic spectra from MS data"
    parser = ArgumentParser(description=prog_desc)

    parser.add_argument('--beamfile', 
                        help='Path to *npy beam positions file', 
                        required=True)
    parser.add_argument('--outbase', 
                        help='Base name for output files', 
                        required=True)
    parser.add_argument('--outdir', 
                        help='Output directory for beam data (def: cwd)', 
                        required=False, default='.')
    parser.add_argument('--uv_taper_lam',
                        help='Lower baseline taper in wavelengths' +\
                             ' (default: 0, no taper)',
                        required=False, type=float, default=0.0)
    parser.add_argument('--field',
                        help='CASA field number of target (def=0)',
                        required=False, type=int, default=0)
    parser.add_argument('--nproc',
                        help='Number of parallel processes',
                        required=False, type=int, default=10)
    parser.add_argument('--no_weights',
                        help='Ignore visibility weights (def: False)',
                        required=False, action='store_true')
    parser.add_argument('--no_flags',
                        help='Ignore MS data flags (def: False)',
                        required=False, action='store_true')
    parser.add_argument('--use_data_col',
                        help='Use DATA column (def: use CORRECTED column)',
                        required=False, action='store_true')
    parser.add_argument('ms_files', nargs='+', 
                        help='Path to data Measurement Set(s). ' +\
                             'If each spw has its own MS, then each should end ' +\
                             'in \'spw[num].ms\'')

    args = parser.parse_args()

    return args


def get_beams(beamfile):
    """
    Make sure beam file exists, then read in coords
    """
    if not os.path.exists(beamfile):
        print(f"Beamfile {beamfile} not found!")
        sys.exit(0)
    beam_list = np.load(beamfile)
    beam_nums = np.arange(len(beam_list), dtype=int)

    print(f"\nFound beamfile: {beamfile}")
    print(f"  with {len(beam_nums)} beams\n")

    return beam_list, beam_nums


def check_outdir(outdir):
    """
    Check that ouput directory exists
    """
    if not os.path.exists(outdir):
        print(f"Output directory {outdir} does not exist!")
        sys.exit(0)
   
    return outdir


def spw_from_fname(mspath):
    """
    Get the spectral window id from ms file name

    Expected to be of the form:

       /some/path/to/basename_spw[id].ms
    """
    # Strip any trailing '/'
    mspath = mspath.rstrip('/')
    
    # Get ms name
    msname = mspath.split('/')[-1]

    # Get num from end of string
    # matching on 'spw_[any length of numbers].ms'  
    m = re.search('spw([0-9]+).ms$', msname)
    
    if m is None:
        print(f"MS {mspath} not in correct form")
        print("   e.g. /path/to/ms_base_spwXXX.ms")
        sys.exit(0)

    spw = int(m.group(1))

    return spw


def check_msfiles(ms_files, target_field=0):
    """
    Make sure the MS data paths exist

    If they do, return ms list and spectral window list
    """
    # Check that all ms files exist
    missing = 0
    for ms_file in ms_files:
        if not os.path.exists(ms_file):
            print(f"MS not found: {ms_file}")
            missing += 1
    if missing:
        sys.exit(0)

    if len(ms_files) == 1:
        msmd.open(ms_files[0])
        spw_list = msmd.spwsforfield(target_field)
        ms_list = ms_files
        msmd.close()

    elif len(ms_files) > 1:
        spws = [ spw_from_fname(ms_file) for ms_file in ms_files ]
        xx = np.argsort(spws)
        spw_list = [spws[ii] for ii in xx]
        ms_list = [ms_files[ii] for ii in xx]
    
    print(f"\nFound {len(ms_list)} MS")
    for mm in ms_list:
        print(mm)

    print(f"\nFound {len(spw_list)} spws")
    print(spw_list)
    
    return ms_files, spw_list



def proc_split_ms(ms_list, spw_list, proc_kwargs):
    """
    Process data with one ms per spw
    """
    basename = proc_kwargs['basename']
    full_freqs = []
    for ii, spw in enumerate(spw_list):
        proc_kwargs['basename'] = f"spw{spw:02d}"
        inms = ms_list[ii]
        mdats, tt, freqs, proc_times = \
                  be.average_visibilities(inms, spws=[0], **proc_kwargs)
        full_freqs.append( freqs )

    # Write freqs
    outdir = proc_kwargs['outdir']
    full_freqs = np.hstack( full_freqs )
    np.save(f"{outdir}/{basename}_freqs.npy", full_freqs)
        
    # Combine spw
    be.multibeam_combine(outdir)

    # Cleanup time arrays
    tdir = f"{outdir}/times"
    if not os.path.exists(tdir):
        os.mkdir(tdir)
    tglob = f"{outdir}/spw*_tt*step*npy"
    tfiles = glob.glob(tglob)
    for tfile in tfiles:
        fname = tfile.split('/')[-1]
        os.rename(tfile, f"{tdir}/{fname}")
    
    return


def proc_one_ms(ms_list, spw_list):
    """
    Process data with many spw in one ms
    """
    return



if __name__ == "__main__":
    # Parse command line input
    args = parse_input()

    # Check and set ouput info
    outdir = check_outdir(args.outdir)
    basename = f"{args.outbase}"
    
    # time tag the log file
    dnow = datetime.now()
    dstr = dnow.strftime('%Y%m%dT%H%M%S') 
    logfile = f"{outdir}/{basename}_{dstr}.log"

    ms_files, spw_list = check_msfiles(args.ms_files, args.field)
    
    # Get beam coordinates and ell em offsets
    beam_locs, beam_nums = get_beams(args.beamfile)
    pos0 = be.get_field_phasecenter(ms_files[0], args.field)
    ell_ems = np.vstack( [be.get_ell_m(pp, pos0) for pp in beam_locs] )

    # Set processing parameters
    # Number of parallel processes
    nproc = args.nproc
    
    # Data column
    if args.use_data_col:
        datacolumn = 'data'
    else:
        datacolumn = 'corrected'
    
    # Target field 
    target_id = args.field

    # Flags?
    if args.no_flags:
        use_flags = False
    else:
        use_flags = True

    # Weights?
    if args.no_weights:
        use_weights = False
    else:
        use_weights = True

    # UV lambda taper?
    uv_lam_taper = args.uv_taper_lam
   
    # Processing parameters 
    proc_kwargs = {'ell_ems' : ell_ems, 
                   'beam_nums' : beam_nums, 
                   'tstep' : 9999.0,
                   'nproc' : nproc,
                   'write_tstep' : 99999,
                   'datacolumn' : datacolumn,
                   'basename' : basename,
                   'target_id' : target_id, 'phase_id' : 3, 'flux_id' : 1,
                   'Nbl_min' : 0,
                   'Nskip' : 0,
                   'use_flags' : use_flags, 
                   'use_weights' : use_weights, 
                   'uv_lam_taper' : uv_lam_taper, 
                   'uvlim' : [1.0, 1e10], 
                   'logfile' : logfile, 
                   'outdir' : outdir}

    if len(ms_files) > 1:
        proc_split_ms(ms_files, spw_list, proc_kwargs)
