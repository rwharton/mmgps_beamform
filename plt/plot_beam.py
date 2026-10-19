import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from argparse import ArgumentParser
import sys, os

#matplotlib.rcParams.update({
#    "text.usetex": True,
#    "font.family": "Helvetica"
#})

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


def stokes_plot(dat_file, freq_file, outfile=None, title=None, use_freqs=None):
    """
    Plot IQUV

    dat_file = npy file with data
    freq_file = npy file with freqs
    """
    dat = np.load(dat_file)
    freqs = np.load(freq_file) / 1e9

    if use_freqs is not None:
        dat = dat[:, :, use_freqs]
        freqs = freqs[use_freqs]
    
    dat *= 1000  # mJy

    I = np.mean(dat[0], axis=0)
    Q = np.mean(dat[1], axis=0)
    U = np.mean(dat[2], axis=0)
    V = np.mean(dat[3], axis=0)

    if outfile is not None:
        plt.ioff()

    fig = plt.figure(figsize=(6, 8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
 

    axI = fig.add_subplot(411)
    axQ = fig.add_subplot(412, sharex=axI)
    axU = fig.add_subplot(413, sharex=axI)
    axV = fig.add_subplot(414, sharex=axI)

    axI.plot(freqs, I, c=colors[0])
    axQ.plot(freqs, Q, c=colors[1])
    axU.plot(freqs, U, c=colors[2])
    axV.plot(freqs, V, c=colors[3])

    frac = 0.95
    pfac = 1.2
    I_ylim = get_yrange(I, frac=frac, pfac=pfac)
    axI.set_ylim(I_ylim)
    
    Q_ylim = get_yrange(Q, frac=frac, pfac=pfac)
    axQ.set_ylim(Q_ylim)
    
    U_ylim = get_yrange(U, frac=frac, pfac=pfac)
    axU.set_ylim(U_ylim)
    
    V_ylim = get_yrange(V, frac=frac, pfac=pfac)
    axV.set_ylim(V_ylim)
    

    axI.set_ylabel("$S_I \\, \\rm{ (mJy)}$", fontsize=14)
    axQ.set_ylabel("$S_Q \\, \\rm{ (mJy)}$", fontsize=14)
    axU.set_ylabel("$S_U \\, \\rm{ (mJy)}$", fontsize=14)
    axV.set_ylabel("$S_V \\, \\rm{ (mJy)}$", fontsize=14)

    tp_kwargs = {'which' : 'major', 'direction': 'in', 'labelbottom' : False, 
                 'top': True, 'bottom': True, 'left' : True, 'right' : True, 
                 'length' : 5} 

    tp_kwargs_bot = tp_kwargs.copy()
    tp_kwargs_bot['labelbottom'] = True

    axI.tick_params(**tp_kwargs)
    axQ.tick_params(**tp_kwargs)
    axU.tick_params(**tp_kwargs)
    axV.tick_params(**tp_kwargs_bot)

    g_kwargs = {'alpha' : 0.3 }
    axI.grid(**g_kwargs)
    axQ.grid(**g_kwargs)
    axU.grid(**g_kwargs)
    axV.grid(**g_kwargs)

    plt.subplots_adjust(hspace=0.0)

    axV.set_xlabel("Frequency (GHz)", fontsize=14)

    if title is not None:
        axI.set_title(title, fontsize=14)

    if outfile is not None:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
        plt.ion()

    else:
        plt.show()

    return 


def time_stokes_plot(dat_file, dt, outfile=None, title=None):
    """
    Plot IQUV

    dat_file = npy file with data
    """
    dat = np.load(dat_file)
    
    dat *= 1000  # mJy

    I = np.mean(dat[0], axis=1)
    Q = np.mean(dat[1], axis=1)
    U = np.mean(dat[2], axis=1)
    V = np.mean(dat[3], axis=1)

    tt = np.arange(len(I)) * dt 

    if outfile is not None:
        plt.ioff()

    fig = plt.figure(figsize=(6, 8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
 

    axI = fig.add_subplot(411)
    axQ = fig.add_subplot(412, sharex=axI)
    axU = fig.add_subplot(413, sharex=axI)
    axV = fig.add_subplot(414, sharex=axI)

    axI.plot(tt,  I, c=colors[0])
    axQ.plot(tt,  Q, c=colors[1])
    axU.plot(tt,  U, c=colors[2])
    axV.plot(tt,  V, c=colors[3])

    axI.set_ylabel("$S_I \\, \\rm{ (mJy)}$", fontsize=14)
    axQ.set_ylabel("$S_Q \\, \\rm{ (mJy)}$", fontsize=14)
    axU.set_ylabel("$S_U \\, \\rm{ (mJy)}$", fontsize=14)
    axV.set_ylabel("$S_V \\, \\rm{ (mJy)}$", fontsize=14)

    tp_kwargs = {'which' : 'major', 'direction': 'in', 'labelbottom' : False, 
                 'top': True, 'bottom': True, 'left' : True, 'right' : True, 
                 'length' : 5} 

    tp_kwargs_bot = tp_kwargs.copy()
    tp_kwargs_bot['labelbottom'] = True

    axI.tick_params(**tp_kwargs)
    axQ.tick_params(**tp_kwargs)
    axU.tick_params(**tp_kwargs)
    axV.tick_params(**tp_kwargs_bot)

    g_kwargs = {'alpha' : 0.3 }
    axI.grid(**g_kwargs)
    axQ.grid(**g_kwargs)
    axU.grid(**g_kwargs)
    axV.grid(**g_kwargs)

    plt.subplots_adjust(hspace=0.0)

    axV.set_xlabel("Time (s)", fontsize=14)

    if title is not None:
        axI.set_title(title, fontsize=14)

    if outfile is not None:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
        plt.ion()

    else:
        plt.show()

    return 


def many_stokes_plots(bnums, freq_file, dt, suffix='full'):
    """
    Make many beam numbers
    """ 
    for bnum in bnums:
        dat_file = f"beam{bnum:03d}_{suffix}.npy"
        ospec = f"IQUV_spec_beam{bnum:03d}.png"
        title = f"beam{bnum:03d}" 
        stokes_plot(dat_file, freq_file, outfile=ospec, title=title)  
        otime = f"IQUV_time_beam{bnum:03d}.png"
        time_stokes_plot(dat_file, dt, outfile=otime, title=title)  

    return



def parse_input():
    """
    Parse arguments to plot_beam
    """
    prog_desc = "Make time and spectrum plots for IQUV data in beams"
    parser = ArgumentParser(description=prog_desc)

    parser.add_argument('--outdir',
                        help='Output directory for plots (def: cwd)',
                        default='.', required=False)
    parser.add_argument('--freqfile',
                        help='npy array of channel frequencies',
                        required=True)
    parser.add_argument('--tsamp', type=float, 
                        help='Visibility time resolution (s)',
                        default=1.0, required=False)
    parser.add_argument('beam_files', nargs='+',
                        help='Beam data file(s) for plotting')

    args = parser.parse_args()

    return args




if __name__ == "__main__":
    # Parse command line input
    args = parse_input()

    freqfile = args.freqfile
    if not os.path.exists(freqfile):
        print(f"Freq file not found: {freqfile}")
        sys.exit(0)

    for bfile in args.beam_files:
        fname = bfile.split('/')[-1]
        fbase = fname.rsplit('.npy', 1)[0]
        ospec = f"{args.outdir}/{fbase}_spec.png"
        otime = f"{args.outdir}/{fbase}_time.png"

        stokes_plot(bfile, freqfile, outfile=ospec, title=fbase)  
        time_stokes_plot(bfile, args.tsamp, outfile=otime, title=fbase)  

