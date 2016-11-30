import sys, os, time, datetime

import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt
import subprocess
import scipy.io.wavfile as wav
import pandas as pd
from IPython.display import display, HTML,Markdown
from scipy import signal
import peakutils



VIDEO_TAG = """<video controls autoplay>
                 <source src="data:video/x-m4v;base64,{0}" type="video/mp4" >
                Your browser does not support the video tag.
                </video>"""

def anim_to_html(anim, fps = 5, fname = 'vid.mp4'):
    if not hasattr(anim, '_encoded_video'):
#         with NamedTemporaryFile(suffix='.mp4') as f:
        with open(fname, 'w') as f:
            anim.save(f.name, fps=fps, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    return VIDEO_TAG.format(anim._encoded_video)

def display_animation(anim, fps = 5, fname = 'vid.mp4'):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim, fps, fname))


def t_to_ind(t, MX_T, MX_S):
    return int(float(t) * MX_S / MX_T)


def ind_to_t(ind, MX_T, MX_S, t0 = 0):
    return (float(ind) / MX_S) * (MX_T - t0)  + t0

def get_windowed_fourier(signal_df, rate, MIN_FREQ = None, MAX_FREQ = None, verbose = False, head = 10, figsz = (12, 8)):

    # Compute the windowed discrete-time Fourier transform of a signal using a sliding window
    freqs, t, Pxx = signal.spectrogram(signal_df['signal'].values, rate)

    # Limit to MIN_FREQ HZ to MAX_FREQ HZ
    if MIN_FREQ:
        Pxx = Pxx[freqs >= MIN_FREQ] 
    if MAX_FREQ:
        Pxx = Pxx[freqs <= MAX_FREQ]

    # Sum over region of frequencies
    fourier_df = pd.DataFrame(Pxx.sum(axis=0), columns = ['signal'])
    fourier_df.index = t
    fourier_df.index.name = 'time (s)'


    if verbose:
        print 'freqs.shape', freqs.shape, 'min:', freqs.min(), 'max', freqs.max()
        print freqs[:head]
        print '\nfourier_df:'
        display(fourier_df.describe())
        display(fourier_df.head(head))
        # Plot with peaks detected
        fourier_df['signal'].plot(figsize=figsz, title ='Transformed Windowed Signal')

    return freqs, fourier_df


def detect_peaks(fourier_df, signal_df, t0 = None, t1 = None, min_thresh = 1900, max_thresh = 500000, head = 10, MIN_DIST = 13, KEY_LEN = 60, back_prop = .3):
    if t0 and t1:
        print 'time range: %.2fs - %.2fs' %(t0,t1)

    sfourier_df = fourier_df.copy()
    ssignal_df = signal_df.copy()
    if t0:
        sfourier_df = sfourier_df[sfourier_df.index >= t0]
        ssignal_df = ssignal_df[ssignal_df.index >= t0]
    if t1:
        sfourier_df = sfourier_df[sfourier_df.index <= t1]
        ssignal_df = ssignal_df[ssignal_df.index <= t1]

    # print 'sfourier_df:'
    # sfourier_df.describe()
    # sfourier_df.head()
    # print 'ssignal_df:'
    # ssignal_df.describe()
    # ssignal_df.head()


    sig = sfourier_df['signal'].values
    # Get indexes withing thresholds
    indexes = peakutils.indexes(sig, min_dist = MIN_DIST, thres=min_thresh / float(sig.max()))
    indexes = indexes[(sig[indexes] <= max_thresh)]


    sfourier_df['is_peak'] = False
    sfourier_df.ix[sfourier_df.index[indexes], 'is_peak'] = True


    # Plot with peaks detected
    _ = plt.figure(figsize = (12,8))
    ax = plt.subplot(111)
    # _ = ax.set_ylim([0, max_thresh*1.1])

    sfourier_df['signal'].plot(ax = ax)

    sfourier_df['signal'].iloc[indexes].plot(style='*', ax=ax)

    print 'Number of Keys detected:', len(indexes)

     # number of milliseconds for a key stroke
    key_len_in_sec = KEY_LEN / 1000.



    peaks = sfourier_df[sfourier_df['is_peak']].copy()
    del peaks['is_peak']
    peaks.index.name = 'peak time'
    peaks = peaks.reset_index()
    peaks['start time'] = peaks['peak time'] - back_prop * key_len_in_sec
    peaks['end time'] = peaks['peak time'] - (1 - back_prop) * key_len_in_sec

    print 'peaks'
    display(peaks.describe())
    display(peaks.head(head))


    return sfourier_df, ssignal_df, peaks

def open_audio(raw_file, verbose = False, head = 5, plt_every = 16, figsz = (12, 8)):


    if not os.path.exists(raw_file):
        print 'file does not exist', raw_file
        return None, None, None

    filename, _ = os.path.splitext(raw_file)
    wav_file = filename + '.wav'
    if not os.path.exists(wav_file):
        print 'Converting %s to %s' % (raw_file, wav_file)
        subprocess.call(['ffmpeg', '-i', raw_file, wav_file])
    
    (rate,sig) = wav.read(wav_file)


    signal_df = pd.DataFrame(sig, columns = ['signal'])
    signal_df.index = signal_df.index.values / float(rate)
    signal_df.index.name = 'time (s)'
    if verbose:
        print '.wav file location:', wav_file
        print 'rate:', rate, 'measurements per second'
        print 'length of audio:', signal_df.index.max(), 'seconds'
        print 'rate * length = ', signal_df.shape[0], 'measurements'
        display(signal_df.head(head))
        signal_df['signal'][::plt_every].plot(title = 'Raw Measurements', figsize= figsz)
    return signal_df, rate, wav_file