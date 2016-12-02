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


import scipy.io.wavfile as wav
from matplotlib import animation
from base64 import b64encode




from scipy.cluster.vq import kmeans2,vq, whiten
from sklearn.cluster import KMeans
from hmmlearn.hmm import MultinomialHMM
from python_speech_features import mfcc


VIDEO_TAG = """<video controls autoplay>
                 <src="data:video/x-m4v;base64,{0}" type="video/mp4" >
                Your browser does not support the video tag.
                </video>"""

def display_animation_from_file(fname):

    return HTML(data = '<video controls alt="test" src="data:video/x-m4v;base64,{0}" autoplay>'.format(b64encode(open(fname, "rb").read())))

def anim_to_html(anim, fps = 5, fname = 'videos/vid.mp4'):
    if not hasattr(anim, '_encoded_video'):
#         with NamedTemporaryFile(suffix='.mp4') as f:
        with open(fname, 'w') as f:
            anim.save(f.name, fps=fps, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    return VIDEO_TAG.format(anim._encoded_video)

def display_animation(anim, fps = 5, fname = 'videos/vid.mp4'):
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
        # display(fourier_df.describe())
        # display(fourier_df.head(head))
        # Plot with peaks detected
        fourier_df['signal'].plot(figsize=figsz, title ='Transformed Windowed Signal')

    return freqs, fourier_df


def detect_peaks(fourier_df, signal_df, t0 = None, t1 = None, min_thresh = 1900, max_thresh = 500000, head = 10, MIN_DIST = 13, KEY_LEN = 60, back_prop = .3, figsz = (10,10), to_add = None):
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

   
    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx
   
        # print [find_nearest(vals, item) for item in to_add]

   

    # Plot with peaks detected
    fig = plt.figure(figsize = figsz)
    ax = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    sig = sfourier_df['signal'].values
    # Get indexes withing thresholds
    indexes = peakutils.indexes(sig, min_dist = MIN_DIST, thres=min_thresh / float(sig.max()))
    indexes = indexes[(sig[indexes] <= max_thresh)]

    if to_add and len(to_add) > 0:
        vals = sfourier_df.index.values
        print to_add
        indexes =  np.append(indexes, [find_nearest(vals, item) for item in to_add])

    sfourier_df['is_peak'] = False
    sfourier_df.ix[sfourier_df.index[indexes], 'is_peak'] = True

    print 'Number of Keys detected:', len(indexes)


    sfourier_df['signal'].plot(ax = ax)
    sfourier_df['signal'].iloc[indexes].plot(style='*', ax=ax, title = 'all')
    ax.axhline(y = min_thresh, linewidth=1, c = 'r')
    
    sfourier_df['signal'].plot(ax = ax2)
    sfourier_df['signal'].iloc[indexes].plot(style='*', ax=ax2, title = 'zoomed')
    ax2.axhline(y = min_thresh, linewidth=1, c = 'r')

    mx = np.max([min_thresh * 1.1, sfourier_df['signal'].max() * .15])
    ax2.set_ylim((0, mx))
    ax3.set_ylim((0, mx))

    # number of milliseconds for a key stroke
    key_len_in_sec = KEY_LEN / 1000.

    peaks = sfourier_df[sfourier_df['is_peak']].copy()
    del peaks['is_peak']
    
    peaks.index.name = 'peak time'
    peaks = peaks.reset_index()
    peaks['start time'] = peaks['peak time'] - back_prop * key_len_in_sec
    peaks['end time'] = peaks['peak time'] + (1 - back_prop) * key_len_in_sec




    if not t0:
        t0 = 0.0

    t1 = t0 + 10.

    sfourier_df = fourier_df.copy()
    ssignal_df = signal_df.copy()
    if t0:
        sfourier_df = sfourier_df[sfourier_df.index >= t0]
        ssignal_df = ssignal_df[ssignal_df.index >= t0]
    if t1:
        sfourier_df = sfourier_df[sfourier_df.index <= t1]
        ssignal_df = ssignal_df[ssignal_df.index <= t1]


    sig = sfourier_df['signal'].values
    # Get indexes withing thresholds
    indexes = peakutils.indexes(sig, min_dist = MIN_DIST, thres=min_thresh / float(sig.max()))
    indexes = indexes[(sig[indexes] <= max_thresh)]



    
    mxt = t1 + 12.
    sfourier_df['signal'].plot(ax = ax3)
    sfourier_df['signal'].iloc[indexes].plot(style='*', ax=ax3, title = 'zoomed beginning')
    ax3.axhline(y = min_thresh, linewidth=1, c = 'r')
    
    

    
    


    return sfourier_df, ssignal_df, peaks

def visualize_clicks(ssignal_df, input_df, peaks, all_peaks, rate, max_thresh, min_thresh = None, outfile = 'videos/output.mp4', MAX_FRAMES = 25., _FRAME_BREAK = 3., SLOWDOWN = 1., figsz = (12,8)):
    global ax, last, FRAME_BREAK, peaks_copy, mainline, ind, line, full_df, time_text
    peaks_copy = peaks.copy()
    FRAME_BREAK = _FRAME_BREAK
    new_rate = int(rate/float(SLOWDOWN))
    wav.write("subset_sound.wav", new_rate, ssignal_df['signal'].values)

    df = input_df.copy()

    TOTAL_TIME = float(df.index[-1] - df.index[0])
    MAX_FRAMES = int(TOTAL_TIME / FRAME_BREAK * MAX_FRAMES)
    SKIPS = df.shape[0] / MAX_FRAMES 
    full_df = df.copy()
    mult = 2**9
    df = df[::SKIPS].copy()

    fig = plt.figure(figsize = figsz)
    ax = plt.axes(ylim = [np.min(np.min(input_df) * .8, 0), min_thresh*10.])
    line, = ax.plot([], [], lw=2)
    mainline = ax.axvline(x = df.index[0], linewidth=2, c = 'k')
    if min_thresh:
        ax.axhline(y = min_thresh, linewidth=3, c = 'k')

    time_text = plt.text(FRAME_BREAK / 2, min_thresh*5, '#START', fontsize=35)


    ind = 0
    last = full_df.index[0] 

    def set_x(update = False):
        global last, full_df, peaks_copy, FRAME_BREAK, ax, time_text, ind
        # Update limits of view
        if update:
            last += FRAME_BREAK
            
        ax.set_xlim(last, last + FRAME_BREAK)
        if update:
            time_text = plt.text((2*last + FRAME_BREAK) / 2., min_thresh*5,getstr(ind-1), fontsize=35)
        
        # Get dfs in view
        df_inds, peak_inds = ((full_df.index >= last) & (full_df.index <= (last + FRAME_BREAK))), ((peaks_copy['peak time'] >= last) & (peaks_copy['peak time'] <= (last + FRAME_BREAK)))
        sub_df, sub_peaks = full_df[df_inds], peaks_copy[peak_inds]
        
        # Plot the signals
        line, = ax.plot(sub_df.index.values, sub_df['signal'].values, 'b', lw=2)
        
        # Plot the clicks
        sub_peaks.apply(lambda x :  ax.axvspan(x['start time'], x['end time'], alpha=0.5, color='red') , axis = 1)
        return line,

    # initialization function: plot the background of each frame
    def init():
        global ax
        line, = set_x()    
        return line,

    def getstr(index):
        try:
            newind = all_peaks[all_peaks['peak time'] == peaks_copy.iloc[index]['peak time']].iloc[0].name
            # print all_peaks.ix[newind]
            st = np.max([0, newind - 2])
            ed = np.min([all_peaks.shape[0], newind + 2])
             
            ret = '%s [%s] %s - %d' % (''.join(all_peaks.ix[st:(newind-1), 'char'].values), all_peaks.ix[newind, 'char'],''.join(all_peaks.ix[(newind+1):ed, 'char'].values),  newind)
            return ret.replace('#SPACE', ' ')
        except : 
            return 'failed'
        

    def animate(i):
        global ind, line, last, mainline, FRAME_BREAK, ax, time_text
        if mainline:
            mainline.set_xdata(df.index[i])
        else:
            mainline = ax.axvline(x = df.index[i], linewidth=1, c = 'k')
        
        if df.index[i] >= (last + FRAME_BREAK):
            line, = set_x(True)
        
        if ind < peaks_copy.shape[0]:
            

            next_peak =  peaks_copy.iloc[ind]['peak time']
            
            while df.index[i] >= next_peak:
                time_text.set_text(getstr(ind))
                if next_peak >= last:
                    ax.axvline(x = next_peak, linewidth=2, c = 'y')
                ind +=1

                if ind < peaks_copy.shape[0]:
                    next_peak =  peaks_copy.iloc[ind]['peak time']
                else:
                    next_peak =  1e15


        return line,

    fps = df.shape[0] / TOTAL_TIME / SLOWDOWN

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames= df.shape[0], interval=1, blit=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'))#, bitrate=new_rate)
    anim.save('videos/im.mp4', writer=writer)
    
    subprocess.call(['ffmpeg', '-i', 'videos/im.mp4', '-i', 'audio/subset_sound.wav', '-c:v', 'copy', '-c:a',
                     'aac', '-strict', 'experimental', outfile, '-y'])

    out = 'videos/' + os.path.splitext(outfile)[0] + '.m4v'
    subprocess.call(['ffmpeg', '-i', outfile, '-vcodec', 'libx264', out, '-y'])
    plt.close(fig)
    print 'done'
    return out




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

    if len(sig.shape) > 1:
        sig = sig[:,0]

    signal_df = pd.DataFrame(sig, columns = ['signal'])
    signal_df.index = signal_df.index.values / float(rate)
    signal_df.index.name = 'time (s)'

    signal_df['signal'] = signal_df['signal'] * 1000. / signal_df['signal'].max()


    if verbose:
        print '.wav file location:', wav_file
        print 'rate:', rate, 'measurements per second'
        print 'length of audio:', signal_df.index.max(), 'seconds'
        print 'rate * length = ', signal_df.shape[0], 'measurements'
        # display(signal_df.head(head))
        signal_df['signal'][::plt_every].plot(title = 'Raw Measurements', figsize= figsz)

    return signal_df, rate, wav_file