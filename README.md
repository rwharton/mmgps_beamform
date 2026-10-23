# mmgps_beamform
A version of ms_beamform specifically designed for MMGPS


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
