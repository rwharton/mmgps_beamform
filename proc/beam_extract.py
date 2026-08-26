import numpy as np
import multiprocessing
import casatools
import time, sys, glob, os, re
from functools import partial
from contextlib import closing
import gc
from astropy.coordinates import SkyCoord
import astropy.units as u

ms = casatools.ms()
msmd = casatools.msmetadata()
qa = casatools.quanta()
me = casatools.measures()

def avg_vis_at_newdir(ell_em, dat, ulams, vlams, wlams):
    """
    Calculate the average visibilities after shifting to a new
    direction given by ell and m, accounting for w-term
    """
    ell, m = ell_em
    n = np.sqrt(1 - ell**2 - m**2)
    dt_phase0 = time.time()
    phase_term = np.exp(-2.0j * np.pi * (ulams * ell + vlams * m + wlams * (n-1)) )
    #phase_term = np.exp(-2.0j * np.pi * (ulams * ell + vlams * m) )
    dt_phase = time.time() - dt_phase0

    dt_mdat0 = time.time()
    mdat = np.sum( (dat * phase_term ), axis=2)
    dt_mdat = time.time() - dt_mdat0
    
    return mdat


def calc_beam(pnum):
    global ell_ems, dat, ulams, vlams, wlams
    partial_avg_vis = partial(avg_vis_at_newdir, dat=dat, 
                              ulams=ulams, vlams=vlams, wlams=wlams)
    one_beam = partial_avg_vis(ell_ems[pnum])
    return (pnum, one_beam)


def initialize_calc_beam(_ell_ems, _dat, _ulams, _vlams, _wlams):
    global ell_ems, dat, ulams, vlams, wlams
    ell_ems = _ell_ems
    dat     = _dat
    ulams   = _ulams
    vlams   = _vlams
    wlams   = _wlams


def calc_all_beams(ell_ems, dat, ulams, vlams, wlams, nproc=4):
    with closing(multiprocessing.Pool(processes=nproc, initializer=initialize_calc_beam, 
                                      initargs = (ell_ems, dat, ulams, vlams, wlams))) as pool:
        results = [pool.apply_async(calc_beam, args=(ii,)) for ii in range(len(ell_ems))]
        print(len(results))
        all_beams = [p.get() for p in results]
    return all_beams


def get_start_stop_chunk(Nt, Nchunk):
    step = int(Nt / Nchunk) + 1
    lovals = []
    hivals = []
    loval = 0
    hival = 0
    for ii in range(Nchunk):
        hival = loval + step
        if hival > Nt:
            hival = Nt
            lovals.append(loval)
            hivals.append(hival)
            break
        else:
            lovals.append(loval)
            hivals.append(hival)
        loval = hival
    return zip(lovals, hivals)


def check_make_dir(dirname):
    """
    Makes new directory called "dirname" if it does't exist.
    If it does exist, does nothing.

    try/except removes race condition (apparently)
    """
    try:
        os.mkdir(dirname)
    except OSError:
        if not os.path.isdir(dirname):
            raise
    return


def save_beam_files(basename, write_step_num, tt, mdats_all, bnums, outdir='.'):
    print(f"    Writing Step {write_step_num}")
    sys.stdout.flush()

    tt_file = f"{outdir}/{basename}_tt_step{write_step_num:03d}.npy"
    
    np.save(tt_file, tt)
    for ii, mdat in enumerate(mdats_all):
        bdir = f"{outdir}/beam{bnums[ii]:05d}"
        check_make_dir(bdir)
        beam_file = f"{basename}_beam{bnums[ii]:05d}_step{write_step_num:03d}.npy"
        beam_path = f"{bdir}/{beam_file}"
        np.save(beam_path, mdat)
    return        


def divide_by_baselines(dd, flags_int, Nbl_min):
    """
    Divide data by number of unflagged baselines 
    per channel 
    """
    Ns = np.sum(flags_int, axis=2, dtype=np.float64)
    #xx = np.where(Ns > 0)
    xx = np.where(Ns > Nbl_min)
    Ns_inv = np.zeros(shape=Ns.shape, dtype=np.float64)
    if np.any(xx):
        Ns_inv[xx] = 1.0 / Ns[xx]
    else: pass
    # Reshape so we can mutliply by data array
    #Ns_inv.shape = (1, Ns.shape[0], 1, Ns.shape[1])
    Ns_inv.shape = (Ns.shape[0], Ns.shape[1], 1, Ns.shape[2])
    Ns_inv = np.concatenate( [Ns_inv] * dd.shape[2], axis=2 )
    #Ns_inv = np.concatenate( [Ns_inv] * dd.shape[0], axis=0 )
    dd *= Ns_inv
    return dd


def average_visibilities(infile, ell_ems=[], beam_nums=[], spws=[0], tstep=1.0, nproc=4, 
                         write_tstep=0, basename="", target_id=2, phase_id=1, flux_id=0, 
                         Nbl_min=0, datacolumn='corrected', Nskip=0, use_flags=True, 
                         use_weights=True, uv_lam_taper=0, uvlim=[1.0, 1e10], 
                         outdir='.', logfile=None):
    # Get channel frequencies using the ms metadata tool
    msmd.open(infile)
    freqs = np.array([])
    for spw in spws:
        freqs = np.hstack( (freqs, msmd.chanfreqs(spw)) )
    all_spw = msmd.spwsforfield(target_id)
    msmd.close()

    #pols = ["XX", "XY", "YX", "YY"]
    pols = ["I", "Q", "U", "V"]
    Npols = len(pols)

    speed_of_light = 299792458.0 # m/s
    waves = speed_of_light / freqs
    inv_waves = freqs / speed_of_light

    # For 2 polarizations..
    #inv_waves_mat = np.vstack( (inv_waves, inv_waves) )
    # For Npols pols
    inv_waves_mat = np.vstack( [inv_waves] * Npols )

    Nchan = len(freqs)
    Nbeams = len(ell_ems)
    Nspw = len(all_spw)

    print(f"{Nchan=}")
    print(f"{Nbeams=}")
    print(f"{Nspw=}")

    # Open MS and average visibilities
    ms.open(infile)

    Tsec = np.diff(ms.range('time')['time'])[0]
    Niter = int(Tsec / tstep) + 1
    #Niter = 60
    print(Niter, Tsec)

    # Default columns : array_id, field_id, data_desc_id, time
    ms.selectinit(datadescid=0, reset=True)
    selection = {'field_id' : [target_id, phase_id], 'uvdist' : uvlim}
    ms.select(items = selection)
    ms.selectpolarization( pols )

    log_output("Selected target_id = %d" %target_id, logfile)
    log_output("Selected phase_id  = %d" %phase_id, logfile)
    log_output("Ignoring flux_id  = %d" %flux_id, logfile)

    ms.iterinit(["TIME"], tstep, adddefaultsortcolumns=False)
    ms.iterorigin()

    if datacolumn == 'data':
        data_col = 'data'
    else:
        data_col = 'corrected_data'

    # Output arrays
    loop_time_start = time.time()
    read_times  = []
    avg_times   = []
    ns_times    = []
    write_times = []
    
    mdats_all = []
    tt_all = np.array([])

    write_step_num = 0

    for jj in range(Niter):
        log_output("Time Step %d / %d" %(jj, Niter), logfile)
        tloop = time.time()

        # Check to see if we are skipping this time step
        if jj < Nskip:
            log_output("  SKIPPING STEP %d / %d" %(jj, Nskip), logfile)

            if ms.iternext():
                continue
            else:
                break
        else:
            pass

        if write_tstep > 0:
            write_step_num = int(jj / write_tstep)
        else:
            pass

        read_time_start = time.time()
        dcols = [data_col, "flag", "time", "u", "v", "w", "weight", "field_id", "data_desc_id"]
        rec = ms.getdata(dcols, ifraxis=True)
        read_times.append(time.time() - read_time_start)
        
        print(rec["u"].shape)
        print(rec["time"].shape)

        # Read in the times
        #tt = np.reshape(rec["time"], (-1, Nspw))[:, 0]
        tt = np.reshape(rec["time"], (-1, Nspw))[:, 0]
        Nt = len(tt)   

        # Read in the array of field_ids
        field_ids = np.reshape(rec['field_id'], (-1, Nspw))[:, 0]

        # Calculate ulam and vlam
        # Need to do this b/c default unit is meters
        # Result should have shape (Npols, Nchan, Nbl, Nt) for 2 polarizations
        ulams = np.multiply.outer(inv_waves_mat, np.reshape(rec["u"], (-1, Nt, Nspw))[:, :, 0])
        vlams = np.multiply.outer(inv_waves_mat, np.reshape(rec["v"], (-1, Nt, Nspw))[:, :, 0])
        wlams = np.multiply.outer(inv_waves_mat, np.reshape(rec["w"], (-1, Nt, Nspw))[:, :, 0])

        # Loop over spectral windows, apply flags, divide by baselines 
        ns_time_start = time.time()
        for kk, spw in enumerate(spws):
            # Get indices of current spw
            xx = np.where(rec['data_desc_id'] == spw)[0]
    
            # Get flags
            flags = rec['flag'][:, :, :, xx]

            # Set up array that will handle the flags, weights, and taper
            ww = np.ones( flags.shape )

            # Convert flags from boolean values (True = FLAG, False = No Flag)
            # to integers (0 = Flag, 1 = No Flag) and apply to ww array, or pass
            if use_flags:
                ww *= (((flags * 1.0) + 1.0) % 2.0)

            # If we want to use the visibility weights, apply them now
            # weights are not done per channel, so have one less axis
            # (Npol, Nbl, Nt)
            if use_weights:
                wts = rec['weight'][:, :, xx]
                ww *= np.reshape(wts, (wts.shape[0], 1, wts.shape[1], wts.shape[2]))

            if uv_lam_taper > 0:
                uv_dist_lams = np.sqrt( ulams**2 + vlams**2 )
                ww_taper = 1 - np.exp( -1 * uv_dist_lams**2 / (2 * uv_lam_taper**2) )
                ww *= ww_taper

            # Multiply by weights
            ddk = rec[data_col][:, :, :, xx] * ww
            
            # Divide data by sum of data weights
            # This was originally unflagged baselines but that's 
            # just weights of 1 or 0
            ddk = divide_by_baselines(ddk, ww, Nbl_min)

            # Extend data along frequency channel axis
            if kk == 0:
                dd = ddk
            else:
                dd = np.concatenate((dd, ddk), axis=1)
        ns_times.append(time.time() - ns_time_start)

        # Get time slices in prep for averaging
        log_output("  Nt = %d, len(tt) = %d, dd.shape[-1] = %d" %(Nt, len(tt), dd.shape[-1]), logfile)
        # Stack the new times
        tt_all = np.hstack( (tt_all, tt) )

        # Calculate mdats if field_id == target_id, otherwise just dump zeros
        avg_time_start = time.time()
        if phase_id in field_ids:
            log_output("  **Blanking out phase-cal field**", logfile)
            sys.stdout.flush()
            #mdats_beams = [ np.zeros( (Nt, Nchan), dtype=dd.dtype ) ] * Nbeams
            mdats_beams = [ np.zeros( (Nt, Nchan), dtype=np.float32 ) ] * Nbeams
        elif flux_id in field_ids:
            log_output("  **Blanking out flux-cal field**", logfile)
            sys.stdout.flush()
            mdats_beams = [ np.zeros( (Nt, Nchan), dtype=np.float32 ) ] * Nbeams
        else:
            mdats = calc_all_beams(ell_ems, dd, ulams, vlams, wlams, nproc=nproc)
            mdats.sort()
            #mdats_beams = [mm[1].T for mm in mdats]
            mdats_beams = [np.real(mm[1].T).astype(np.float32) for mm in mdats]
        avg_times.append(time.time() - avg_time_start)

        # If mdats_all is empty (ie, this is step 0 or after a write), then 
        # fill it directly with the beams
        if len(mdats_all) == 0:
            mdats_all = [ mdats_beams[ii] for ii in range(Nbeams) ]
        # Otherwise stack on to previous data
        else:
            mdats_all = [ np.vstack((mdats_all[ii], mdats_beams[ii])) for ii in range(Nbeams) ]

        # If this is a write step, write out numpy arrays
        if write_tstep > 0:
            if jj == ((write_step_num + 1) * write_tstep - 1):
                write_time_start = time.time()
                save_beam_files(basename, write_step_num, tt_all, 
                                mdats_all, beam_nums, outdir=outdir)
                mdats_all = []
                tt_all = np.array([])
                write_times.append( time.time() - write_time_start )
            
        dtloop = time.time() - tloop
        log_output("   Step %d proc time: %.2f min" %(jj, dtloop / 60.0), logfile)
        # Go on to next iteration if it's there
        if ms.iternext():
            pass
        else:
            break
    # If we are writing data to file and there is something left, 
    # write that now.
    if len(mdats_all) and write_tstep > 0:
        write_time_start = time.time()
        save_beam_files(basename, write_step_num, tt_all, mdats_all, 
                        beam_nums, outdir=outdir)
        mdats_all = []
        tt_all = np.array([])
        write_times.append( time.time() - write_time_start )

    loop_time = time.time() - loop_time_start

    ms.iterend()
    ms.close()

    # Cut out the dummy row for data arrays
    #mdats_all = [ mm[1:] for mm in mdats_all ]

    log_output("LOOP TIME   = %.2f minutes" %(loop_time / 60.0), logfile)
    log_output("READ TIME   = %.2f minutes" %(np.sum(read_times) / 60.0), logfile)
    log_output("WRITE TIME  = %.2f minutes" %(np.sum(write_times) / 60.0), logfile)
    log_output("VISAVG TIME = %.2f minutes" %(np.sum(avg_times) / 60.0), logfile)
    log_output("DENOM TIME  = %.2f minutes" %(np.sum(ns_times) / 60.0), logfile)

    proc_times = [read_times, write_times, avg_times, ns_times]

    return mdats_all, tt_all, freqs, proc_times



def data_cross_check(t0, freqs0, dat0, t1, freqs1, dat1):
    if len(t0) != len(t1):
        print("Time mismatch!")
        return 0
    elif len(set(np.diff(freqs0))) > 1  or len(set(np.diff(freqs1))) > 1:
        print("Multiple channel widths!")
        return 0
    elif np.max( np.abs( t1 - t0 ) ) > 0:
        print("Time Offset!")
        return 0
    else:
        return 1


def dir_to_coord(dd):
    """
    Convert a casa "direction" to a position like 

    ["J2000", "03:58:53.716501", "+54.13.13.72701"]
    """
    ra_rad  = dd['m0']['value']
    dec_rad = dd['m1']['value']

    cc = SkyCoord(ra_rad, dec_rad, unit=(u.rad, u.rad), frame='fk5')
    
    ra_str, dec_str = cc.fk5.to_string('hmsdms').split()
    """
    ra_str = ra_str.replace('h', ':')
    ra_str = ra_str.replace('m', ':')
    ra_str = ra_str.replace('s', '')

    dec_str = dec_str.replace('d', '.')
    dec_str = dec_str.replace('m', '.')
    dec_str = dec_str.replace('s', '')
    """
    bcenter = ["J2000", ra_str, dec_str]

    return bcenter


def get_field_phasecenter(msfile, fieldid=0):
    """
    Open ms, get phase center of field fieldid 
    and write out as coord string
    """
    ms.open(msfile)
    dd = ms.getfielddirmeas(fieldid=fieldid)
    bcenter = dir_to_coord(dd)
    ms.close()
    return bcenter


def get_ell_m(pos1, pos0):
    """
    Input positions are expected to be in the form
    [ref-frame (string), RA (string), DEC (string)]
    where each is in a valid casa format.

    Example:

    ["J2000", "03:58:53.716501", "+54.13.13.72701"]
    
    """
    
    dir0 = me.direction(pos0[0], pos0[1], pos0[2])
    dir1 = me.direction(pos1[0], pos1[1], pos1[2])

    """
    dir1_ra_fix  = me.direction(pos1[0], pos1[1], pos0[2])
    dir1_dec_fix = me.direction(pos1[0], pos0[1], pos1[2])

    ra_sep  = me.separation(dir1_ra_fix, dir0)
    dec_sep = me.separation(dir1_dec_fix, dir0)

    sgn_ell = +1.0 if qa.gt(dir1['m0'], dir0['m0']) else -1.0
    sgn_m   = +1.0 if qa.gt(dir1['m1'], dir0['m1']) else -1.0

    ell = sgn_ell * qa.getvalue(qa.sin(ra_sep))[0]
    m   = sgn_m  * qa.getvalue(qa.sin(dec_sep))[0]
    """
    ra0  = dir0['m0']
    dec0 = dir0['m1']
    
    ra   = dir1['m0']
    dec  = dir1['m1']
    dra  = qa.sub(ra, ra0)

    cos_dec  = qa.getvalue( qa.cos( dec ) )
    cos_dec0 = qa.getvalue( qa.cos( dec0 ) )
    
    sin_dec  = qa.getvalue( qa.sin( dec ) )
    sin_dec0 = qa.getvalue( qa.sin( dec0 ) )
    
    sin_dra  = qa.getvalue( qa.sin( dra ) )
    cos_dra  = qa.getvalue( qa.cos( dra ) )

    ell = cos_dec * sin_dra 
    m   = sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_dra

    ell = ell[0]
    m = m[0] 

    return ell, m


def get_time_info(tt):
    """
    tt : Time in MJD seconds
    """
    tt_tu = {'value' : tt, 'unit' : 's'}
    tt_dict = qa.splitdate(tt_tu)
    mjd = tt_dict['mjd']
    stt_imjd = int( mjd )
    stt_smjd = int((mjd - int(mjd)) * 24 * 3600.0)
    stt_offs = (mjd - int(mjd)) * 24 * 3600.0 - stt_smjd
    fits_date = qa.time(tt_tu, form='fits')
    out_dict = {'obs_date' : fits_date, 
                'stt_imjd' : stt_imjd, 
                'stt_smjd' : stt_smjd,
                'stt_offs' : stt_offs, 
                'mjd'      : mjd}
    return out_dict


def get_and_save_spw_freqs(infile, spw_list, basename=""):
    """
    Save freqs in MHz
    """
    for spw in spw_list:
        msmd.open(infile)
        freqs = msmd.chanfreqs(spw)
        msmd.close()

        outfile = "%s_freqs_spw%02d.npy" %(basename, spw)
        np.save(outfile, freqs / 1e6)
    return


def double_check_freqs(infile, spw_list, tstep=0.01, field=2):
    freqlist_msmd = []
    freqlist_dat  = []
    
    for spw in spw_list:    
        # Get channel frequencies using the ms metadata tool
        msmd.open(infile)
        freqs_msmd = msmd.chanfreqs(spw) / 1e6
        msmd.close()
        freqlist_msmd.append(freqs_msmd)
        
        # Get frequencies from ms
        ms.open(infile)
        ms.selectinit(datadescid=spw)
        ms.select({'field_id' : field})
        ms.iterinit(["TIME"], tstep, adddefaultsortcolumns=False)
        ms.iterorigin()
        data_col = 'corrected_data'
        rec = ms.getdata([data_col, "axis_info"], ifraxis=True)
        freqs_dat = rec["axis_info"]["freq_axis"]["chan_freq"].ravel() / 1e6
        freqlist_dat.append(freqs_dat)
        
    return freqlist_msmd, freqlist_dat        


def log_output(text, logfile):
    if logfile is None:
        print(text)
        sys.stdout.flush()
    else:
        fout = open(logfile, 'a+')
        fout.write("%s\n" %text)
        fout.close()
    return


def combine_spw(indir, bnum, outbase=None):
    dd_list = []
    # get spws 
    glob_str = f"{indir}/spw*_beam{bnum:05d}*npy"
    npyfiles = glob.glob(glob_str)
    spw_list = []
    for nfile in npyfiles:
        fname = nfile.split('/')[-1]
        p = re.search(f"spw([0-9]+)_beam{bnum:05d}", fname)
        if p is not None:
            spw_list.append( int(p.group(1)) )
    
    spws = np.unique(spw_list)

    for ii in spws:
        infile = f"{indir}/spw{ii:02d}_beam{bnum:05d}_step000.npy"
        dd_ii = np.load(infile)
        dd_list.append(dd_ii)

    dd0 = np.hstack( [ddi[:, :, 0] for ddi in dd_list ] )
    dd1 = np.hstack( [ddi[:, :, 1] for ddi in dd_list ] )
    dd2 = np.hstack( [ddi[:, :, 2] for ddi in dd_list ] )
    dd3 = np.hstack( [ddi[:, :, 3] for ddi in dd_list ] )

    dd_out = np.array([dd0, dd1, dd2, dd3])

    if outbase is None:
        outfile = f"{indir}/beam{bnum:05d}_full.npy" 
    else:
        outfile = f"{indir}/{outbase}_beam{bnum:05d}_full.npy" 
        
    np.save(outfile, dd_out)

    # If everything worked, then clean up the spw files
    if os.path.exists(outfile):
        for nfile in npyfiles:
            if os.path.exists(nfile):
                os.remove(nfile)

    return


def multibeam_combine(topdir, outbase=None):
    """
    assume b00000 etc
    """
    glob_str = f"{topdir}/beam[0-9]*[0-9]"
    bdirs = glob.glob(glob_str)
    bnum_list = [ int(bb.rsplit('beam', 1)[-1]) for bb in bdirs ]
    bnums = np.unique(bnum_list)
    for bnum in bnums:
        indir = f"{topdir}/beam{bnum:05d}"
        combine_spw(indir, bnum, outbase=outbase)
    return

