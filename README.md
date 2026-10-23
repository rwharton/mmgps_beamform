# mmgps_beamform
A visibility-domain pipeline for measuring the polarisation properties of sources in MMGPS data.

## Preliminaries and Prerequisites

This software is specifically tailored to processing MMGPS data, but may be applied to other data sets.  The assumed data structure is a set of CASA Measurement Sets (MS) where each spectral window has its own MS (e.g., `MSGPS_L_1184_spw004.ms`).  It also requires a source catalog in FITS format that is (mostly) a pyBDSF catalog.

## Run beamformer to extract IQUV spectra


```
usage: run_beamform.py [-h] --catfile CATFILE [--cat_max_sep CAT_MAX_SEP] [--cat_isl_filter] --outbase OUTBASE
                       [--outdir OUTDIR] [--nproc NPROC] [--uv_taper_lam UV_TAPER_LAM] [--field FIELD] [--no_weights]
                       [--no_flags] [--use_data_col]
                       ms_files [ms_files ...]

Extract beamformed IQUV dynamic spectra from MS data

positional arguments:
  ms_files              Path to data Measurement Set(s). If each spw has its own MS, then each should end in 'spw[num].ms'

optional arguments:
  -h, --help            show this help message and exit
  --catfile CATFILE     Path to input source catalog
  --cat_max_sep CAT_MAX_SEP
                        Pre-filter: Max sep from phase center in arcmin (default: -1, no limit)
  --cat_isl_filter      Pre-filter: Only keep one source per island (def: do NOT filter)
  --outbase OUTBASE     Base name for output files
  --outdir OUTDIR       Output directory for beam data (def: cwd)
  --nproc NPROC         Number of parallel processes
  --uv_taper_lam UV_TAPER_LAM
                        Lower baseline taper in wavelengths (default: 0, no taper)
  --field FIELD         CASA field number of target (def=0)
  --no_weights          Ignore visibility weights (def: False)
  --no_flags            Ignore MS data flags (def: False)
  --use_data_col        Use DATA column (def: use CORRECTED column)
```

## Run RM-Sythesis and RM-CLEAN 

```
usage: rm_proc_beams.py [-h] [--outdir OUTDIR] --freqfile FREQFILE [--maskfile MASKFILE] [--keep_zeros] [--niter NITER]
                        [--threshold THRESHOLD] [--phimax PHIMAX] --catfile CATFILE [--use_obs_errs] [--nproc NPROC]
                        dat_files [dat_files ...]

Make time and spectrum plots for IQUV data in beams

positional arguments:
  dat_files             Beam data file(s) for RM CLEAN-ing

options:
  -h, --help            show this help message and exit
  --outdir OUTDIR       Output directory (def: cwd)
  --freqfile FREQFILE   npy array of channel frequencies in MHz
  --maskfile MASKFILE   npy array of channel mask (1=use chan, 0=skip chan)
  --keep_zeros          Do NOT flag zeros in spectrum (default = False)
  --niter NITER         Number of CLEAN iterations (def=200)
  --threshold THRESHOLD
                        CLEAN threshold in (neg) sigma (def=-2)
  --phimax PHIMAX       Maximum Faraday depth (rad/^2, def: -1, no max)
  --catfile CATFILE     FITS catalog file to add results to
  --use_obs_errs        Use obs errors from RM-tools in catalog (default = False)
  --nproc NPROC         Number of parallel processes to run (def=1)
```

## Make Pretty Plots 


```
usage: rm_diagnostic_plots.py [-h] [--outdir OUTDIR] --catfile CATFILE --imfile IMFILE [--snr_cut SNR_CUT]
                              [--outbase OUTBASE] --rm_spec_dir RM_SPEC_DIR [--bfmt BFMT] [--phi_win PHI_WIN]
                              [--imsize IMSIZE] [--no_pfrac] [--no_cutouts] [--no_cutouts_select] [--no_rmplot]

Make time and spectrum plots for IQUV data in beams

options:
  -h, --help            show this help message and exit
  --outdir OUTDIR       Output directory for plots (def: cwd)
  --catfile CATFILE     FITS catalog of beam sources
  --imfile IMFILE       FITS Image of field
  --snr_cut SNR_CUT     Minimum PI SNR
  --outbase OUTBASE     Base of output files
  --rm_spec_dir RM_SPEC_DIR
                        Top of beam rm spec directory
  --bfmt BFMT           fstring format of beam name
  --phi_win PHI_WIN     Size of RM Spec zoom panel in units of RMSF (def:10 * RMSF)
  --imsize IMSIZE       Size of RM sky plot in degrees (def:1.5)
  --no_pfrac            Do NOT make pol frac plot
  --no_cutouts          Do NOT make cutout plots for ALL sources
  --no_cutouts_select   Do NOT make cutout plots for THRESHOLD sources
  --no_rmplot           Do NOT RM image plot
```
