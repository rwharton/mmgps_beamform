import numpy as np
import time
import sys
import glob
import npy2psrfits as npsr 
import os
import re
import multiprocessing
from contextlib import closing


def data_cross_check(t0, freqs0, dat0, t1, freqs1, dat1):
    if len(t0) != len(t1):
        print("Time mismatch!")
        return 0
    elif len(t0) != len(dat0):
        print("Time / Data Row Mismatch (0)")
    elif len(t1) != len(dat1):
        print("Time / Data Row Mismatch (1)")
    elif len(set(np.diff(freqs0))) > 1  or len(set(np.diff(freqs1))) > 1:
        print("Multiple channel widths!")
        return 0
    elif np.max( np.abs( t1 - t0 ) ) > 0:
        print("Time Offset!")
        return 0
    else:
        return 1


def combine_spws(t0, freqs0, dat0, t1, freqs1, dat1):
    if data_cross_check(t0, freqs0, dat0, t1, freqs1, dat1):
        pass
    else:
        return
    
    freq_lo = np.min( np.hstack((freqs0, freqs1)) )
    freq_hi = np.max( np.hstack((freqs0, freqs1)) )
    df = np.abs( np.diff(freqs0) )[0]
    all_freqs = np.arange(freq_lo, freq_hi + df, df)

    N0 = len(freqs0)
    N1 = len(freqs1)
    Nf = len(all_freqs)
    Nt = len(t0)

    full_dat = np.zeros( (Nt, Nf), dtype=dat0.dtype)

    x0 = np.where(all_freqs == freqs0[0])[0][0]
    x1 = np.where(all_freqs == freqs1[0])[0][0]

    full_dat[:, x0:x0+N0] = dat0[:, :]
    full_dat[:, x1:x1+N1] = dat1[:, :]

    tt = np.copy(t0)

    return tt, all_freqs, full_dat


def combine_beam(bfile_list, outname=None):
    """
    Concatenate all data chunks for a given beam.
    If an outname is given, the array will be saved 
    to file, otherwise it is just returned.
    """
    bfile_list.sort()
    dat_list = []
    for bfile in bfile_list:
        dd = np.load(bfile)
        dat_list.append(dd)
    all_dat = np.concatenate(dat_list, axis=0)
    if outname is None:
        return all_dat
    else:
        np.save(outname, all_dat)
        return 


def combine_one_fullband_beam(basename, beam_num):
    glob_str  = "%s_beam%05d_step*.npy" %(basename, beam_num)
    beam_files = glob.glob(glob_str)

    print("  Found %d files for beam %d" %(len(beam_files), beam_num))

    dat = combine_beam(beam_files)
    tt = combine_tts(basename)

    return tt, dat


def combine_one_fullband_beam_fromlist(beam_files, beam_num):
    print("  Combining %d files for beam %d" %(len(beam_files), beam_num))
    dat = combine_beam(beam_files)

    return dat


def get_step_nums(basename):
    tt_files = glob.glob("%s_tt_step*.npy" %(basename))
    tt_files.sort()
    tnums = np.array([ ttstr.split("step")[-1].rstrip(".npy") \
                           for ttstr in tt_files ]).astype('int')
    print("  Found %d steps -- (%d, %d)" %(len(tnums), tnums[0], tnums[-1]))
    return tnums


def beam2fits(bnum, step_nums, Nskip, full_freqs, yy, basename, src_base, 
              bnum_list, beam_list, use_real, out_type, keep_npy, 
              obs_params):
    # Beam string (will use this a lot)
    bstr = "beam%05d" %(bnum)
    print("Working on %s" %bstr)

    # Step files are in beamXXXX sub-directories
    step_files = ["%s/%s_%s_step%03d.npy" %(bstr, basename, bstr, snum) \
                      for snum in step_nums ]

    tmp_dat = combine_one_fullband_beam_fromlist(step_files, bnum)
    dat = np.zeros( (len(tmp_dat), len(full_freqs)) )
    dat[:, yy] = tmp_dat[:]
    
    if use_real:
        dat = np.real(dat)
    else: pass

    if out_type is not None:
        dat = dat.astype(out_type)
    else: pass

    # Skipping Nsec seconds in data
    if Nskip > 0:
        dat = dat[Nskip:]
        print("\nSkipping %.2f seconds !!!\n" %(Nskip * obs_params['dt']))
    else:
        pass

    npy_file = "%s/%s_%s.npy" %(bstr, basename, bstr)
    print("...write to %s" %npy_file)
    np.save(npy_file, dat)
    print("...done.")

    outname = "%s_%s" %(basename, bstr)
    src_name, ra_str, dec_str = source_info_from_beamlist(beam_list, bnum, 
                                                          src_base)

    print("BEAM%05d" %bnum)
    print("  %s" %npy_file)
    print("  %s" %outname)
    print("  (%s, %s, %s)" %(src_name, ra_str, dec_str))

    npsr.convert_npy_to_psrfits(npy_file, outname, src_name=src_name, 
                                ra_str=ra_str, dec_str=dec_str, 
                                **obs_params)

    if keep_npy:
        pass
    else:
        os.remove(npy_file)
    return


def beam2fits_multi(nproc, group_bnums, step_nums, Nskip, full_freqs, yy,
                    basename, src_base, bnum_list, beam_list, use_real, 
                    out_type, keep_npy, obs_params):

    with closing(multiprocessing.Pool(processes=nproc)) as pool:

        results = [pool.apply_async(beam2fits, 
                   args=(bnum, step_nums, Nskip, full_freqs, yy, basename, 
                         src_base, bnum_list, beam_list, use_real, 
                         out_type, keep_npy, obs_params)\
                                   ) for bnum in group_bnums]
            
        all_beams = [p.get() for p in results]
    return


def get_beam_groups(bnum_list, n):
    Nb = len(bnum_list)
    beam_groups = [ bnum_list[ii : ii + n] for ii in range(0, Nb, n) ]
    return beam_groups


def combine_beams_to_psrfits(basename, src_base, bnum_list, beam_list,
                             use_real=True, out_type=None, 
                             keep_npy=False, obs_params={}, 
                             skip_sec=0.0, nproc=1):

    step_nums = get_step_nums(basename)

    freqs = get_all_freqs(basename)
    full_freqs, yy = get_freq_mapping(freqs)
    print("Freqs: (%.2f, %.2f, %.2f) -- %d" %(\
            full_freqs[0], full_freqs[-1], 
            np.diff(full_freqs)[0], len(full_freqs)))

    # Skipping Nsec seconds
    dt = obs_params['dt']
    if skip_sec > 0:
        Nskip = int(skip_sec / dt)
        obs_params['mjd_start'] += (Nskip * dt) / (24 * 3600.0)
    else:
        Nskip = 0

    beam_groups = get_beam_groups(bnum_list, nproc)

    for group_bb in beam_groups:
        print(group_bb)
        beam2fits_multi(nproc, group_bb, step_nums, Nskip, full_freqs, yy,
                        basename, src_base, bnum_list, beam_list, 
                        use_real=use_real, out_type=out_type, 
                        keep_npy=keep_npy, obs_params=obs_params)
    return
    

def combine_tts(basename_in, basename_out=None):
    glob_str = "%s_tt_step*npy" %(basename_in)
    tt_list = glob.glob(glob_str)
    tt_list.sort()
    tt_dats = [ np.load(tti) for tti in tt_list ]
    tt = np.hstack(tt_dats)
    if basename_out is None:
        return tt
    else:
        outname = "%s_tt.npy" %(basename_out)
        np.save(outname, tt)
        return outname


def source_info_from_beamlist(beamlist, bnum, src_base):
    epoch_str, ra_str, dec_str = beamlist[bnum]
    # Convert from hms/dms to :
    if ra_str.count("h") > 0:
        ra_hval, ra_mval, ra_sval = re.split('[hms]', ra_str)[:3]
        dec_dval, dec_mval, dec_sval = re.split('[dms]', dec_str)[:3]
        ra_str = "%02d:%02d:%08.5f" %(int(ra_hval), int(ra_mval), 
                                      float(ra_sval))
        dec_str = "%+02d:%02d:%08.5f" %(int(dec_dval), int(dec_mval), 
                                        float(dec_sval))
    elif dec_str.count(":") == 0:
        dec_str = dec_str.replace('.', ':', 2)
    else:
        pass
        
    name_str = "%s_BEAM%05d" %(src_base, bnum)
    return name_str, ra_str, dec_str


def rename_spw_files(file_list):
    for old_name in file_list:
        old_beg, old_end = old_name.split("_spw0_", 1)
        new_name = "%s_%s" %(old_beg, old_end)
        print("%s   -->   %s" %(old_name, new_name))
        os.rename(old_name, new_name)
    return


def rename_beam_files(file_list):
    for old_name in file_list:
        old_beg, old_end = old_name.split("0_", 1)
        new_name = "%s_%s" %(old_beg, old_end)
        print("%s   -->   %s" %(old_name, new_name))
        os.rename(old_name, new_name)
    return


def rename_beam_files2(file_list):
    for old_name in file_list:
        old_beg, old_end = old_name.split("_test_", 1)
        new_name = "%s_%s" %(old_beg, old_end)
        print("%s   -->   %s" %(old_name, new_name))
        os.rename(old_name, new_name)
    return
    

def get_freqs_and_times(basename):
    tt = get_all_times(basename)
    freqs = get_all_freqs(basename)
    return tt, freqs


def get_all_freqs(basename):
    freq_files = glob.glob("%s_freqs_spw*.npy" %(basename))
    freq_files.sort()
    freqs = np.array([])
    
    for ff_file in freq_files:
        ffi = np.load(ff_file)
        freqs = np.hstack( (freqs, ffi) )
        
    return freqs
    

def get_all_times(basename):
    time_files = glob.glob("%s_tt_step*.npy" %(basename))
    time_files.sort()
    tt = np.array([])
    
    for tt_file in time_files:
        tti = np.load(tt_file)
        tt = np.hstack( (tt, tti) )
    
    return tt 


def get_freq_mapping(freqs):
    dfreqs = np.diff(freqs)
    df = np.min( np.abs(dfreqs)[ np.abs(dfreqs) > 0 ] )
    print(df)
    f_lo = np.min(freqs)
    f_hi = np.max(freqs)
    full_freqs = np.arange(f_lo, f_hi + df, df)
    yy = []
    for ff in freqs:
        jj = np.argmin( np.abs(ff - full_freqs) )
        if np.abs(ff - full_freqs[jj]) < 1e-3:
            yy.append(jj)
    yy = np.array(yy)
    return full_freqs, yy
