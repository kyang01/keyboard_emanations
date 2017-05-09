'''

    signal_processing.py


    This library contains helper functions for the preprocessing and visualizations
    of keystroke data, this includes 

        - Functions to open and convert audio files 

        - Functions to analyze the raw signal

        - Functions to transform signal to fourier space 

        - Functions to build the input for the analysis portion 

        - Functions to overlay signal video with audio and labeled keystrokes  
'''

# Core
import sys, os, time, datetime, subprocess
from base64 import b64encode

# Numpy and pandas
import numpy as np
import pandas as pd

# Matplotlib
from matplotlib import mlab
import matplotlib.pyplot as plt
from matplotlib import animation

# For opening wav files 
import scipy.io.wavfile as wav
from scipy import signal

# For displaying in notebook
# from IPython.display import display, HTML,Markdown

# For detecting peaks
import peakutils


# Video tag to embed in ipython notebook
VIDEO_TAG = """<video controls autoplay>
                 <src="data:video/x-m4v;base64,{0}" type="video/mp4" >
                Your browser does not support the video tag.
                </video>"""

def display_animation_from_file(fname):
    '''
        Displays aiv file, fname, in notebook

            fname : file location of aiv file 
    '''

    return HTML(data = '<video controls alt="test" src="data:video/x-m4v;base64,{0}" autoplay>'.format(b64encode(open(fname, "rb").read())))

def anim_to_html(anim, fps = 5, fname = 'videos/vid.mp4'):
    '''
        Converts an animation into html video tag, saves to file etc 

            fps : Frames per second of video

            fname : Location to store video 
    '''
    if not hasattr(anim, '_encoded_video'):
        with open(fname, 'w') as f:
            anim.save(f.name, fps=fps, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    return VIDEO_TAG.format(anim._encoded_video)

def display_animation(anim, fps = 5, fname = 'videos/vid.mp4'):
    '''
        Displays animation in notebook 

            fps : Frames per second of video

            fname : Location to store video 
    '''
    plt.close(anim._fig)
    return HTML(anim_to_html(anim, fps, fname))


def get_windowed_fourier(signal_df, rate, MIN_FREQ = None, MAX_FREQ = None, verbose = False, 
                                    figsz = (12, 8)):
    '''
        Get the windowed fourier signal of the raw audio 

            signal_df : Dataframe of the raw signal 

            rate : rate at which signal was read in 

            MIN_FREQ : If has a value, only frequencies above this range are kept 

            MAX_FREQ : IF has a value, only frequencies below this range are kepy 

            verbose : Whether or not to print out 

            figsz : Size of the plot generated if verbose 
    '''

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
        print('freqs.shape', freqs.shape, 'min:', freqs.min(), 'max', freqs.max())

        # Plot with peaks detected
        fourier_df['signal'].plot(figsize=figsz, title ='Transformed Windowed Signal')

    return freqs, fourier_df


def detect_peaks(fourier_df, signal_df, t0 = None, t1 = None, min_thresh = 1900, 
                max_thresh = 500000, min_dist = 13, key_len = 60, back_prop = .3, 
                figsz = (10,10), to_add = None):
    '''
        Detect Peaks in the fourier_df 

            fourier_df : Dataframe containing transformed fourier signal 

            t0 : The start time to analyze 

            t1 : The end time to analyze 

            min_thresh : The minimum threshold to include as a key press 

            max_thresh : The maximium threshold value to include as a key press 

            min_dist : The minimum distanec allowed between two peaks 

            key_len : The fixed length of a key press in milliseconds 

            back_prop : The key_len of a key press is started at back_prop * key_len 
                and ends at back_prop * (1-key_len)

            figsz : The size of the plot

            to_add : If not None, is a list of indices to add as peaks 

    '''
    if t0 and t1:
        print('time range: %.2fs - %.2fs' %(t0,t1))

    # number of milliseconds for a key stroke
    key_len_in_sec = key_len / 1000.

    # Copy over dataframes
    sfourier_df = fourier_df.copy()
    ssignal_df = signal_df.copy()

    # Restrict to timeperiod
    if t0:
        sfourier_df = sfourier_df[sfourier_df.index >= t0]
        ssignal_df = ssignal_df[ssignal_df.index >= t0]
    if t1:
        sfourier_df = sfourier_df[sfourier_df.index <= t1]
        ssignal_df = ssignal_df[ssignal_df.index <= t1]

    
    # Finds the index of the closest value in an array
    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx
      
    # Plot with peaks detected
    fig = plt.figure(figsize = figsz)
    ax = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    # The fourier signal
    sig = sfourier_df['signal'].values

    # Get indexes withing thresholds
    indexes = peakutils.indexes(sig, min_dist = min_dist, thres=min_thresh / float(sig.max()))
    indexes = indexes[(sig[indexes] <= max_thresh)]

    # Add on additional indices if specified 
    if to_add and len(to_add) > 0:
        vals = sfourier_df.index.values
        indexes =  np.append(indexes, [find_nearest(vals, item) for item in to_add])

    # Add on column to indicate if time point is a peak
    sfourier_df['is_peak'] = False
    sfourier_df.ix[sfourier_df.index[indexes], 'is_peak'] = True


    # Create dataframe of peaks
    peaks = sfourier_df[sfourier_df['is_peak']].copy()
    del peaks['is_peak']
    peaks.index.name = 'peak time'
    peaks = peaks.reset_index()

    peaks['start time'] = peaks['peak time'] - back_prop * key_len_in_sec
    peaks['end time'] = peaks['peak time'] + (1 - back_prop) * key_len_in_sec

    print('Number of Keys detected:', len(indexes))

    # Plot the entire signal with peaks
    sfourier_df['signal'].plot(ax = ax)
    sfourier_df['signal'].iloc[indexes].plot(style='*', ax=ax, title = 'all')
    ax.axhline(y = min_thresh, linewidth=1, c = 'r')
    
    # Plot the entire signal zoomed in  on the threshold
    sfourier_df['signal'].plot(ax = ax2)
    sfourier_df['signal'].iloc[indexes].plot(style='*', ax=ax2, title = 'zoomed')
    ax2.axhline(y = min_thresh, linewidth=1, c = 'r')

    # Change the threshold 
    mx = np.max([min_thresh * 1.1, sfourier_df['signal'].max() * .15])
    ax2.set_ylim((0, mx))

    # Plot a shortened time period
    if not t0:
        t0 = 0.0
    t1 = t0 + 10.

    if t0:
        fourier_df = fourier_df[fourier_df.index >= t0]
        signal_df = signal_df[signal_df.index >= t0]
    if t1:
        fourier_df = fourier_df[fourier_df.index <= t1]
        signal_df = signal_df[signal_df.index <= t1]

    # Get signal during shortened period
    sig = fourier_df['signal'].values
    
    # Get indexes withing thresholds
    indexes = peakutils.indexes(sig, min_dist = min_dist, thres=min_thresh / float(sig.max()))
    indexes = indexes[(sig[indexes] <= max_thresh)]
    
    # Plot first 10 seconds of clip
    fourier_df['signal'].plot(ax = ax3)
    fourier_df['signal'].iloc[indexes].plot(style='*', ax=ax3, title = 'zoomed beginning')
    ax3.axhline(y = min_thresh, linewidth=1, c = 'r')
    ax3.set_ylim((0, mx))
    
    return sfourier_df, ssignal_df, peaks

def visualize_clicks(ssignal_df, input_df, peaks, all_peaks, rate, min_thresh, 
                    outfile = 'videos/output.mp4', MAX_FRAMES = 25., _FRAME_BREAK = 3., 
                    SLOWDOWN = 1., figsz = (12,8), mult = 2**9):
    '''
        Function to visualize clicks as an animated video created with matplotlib.animation 
        Combines the audio, with a video of the detected peaks 

            ssignal_df : Dataframe of raw signal during time period 

            intput_df : The dataframe used to plot the signal (signal_df or fourier_df)

            peaks : The peaks in this time period 

            all_peaks : All of the peaks in the entire video 

            rate : The rate to save the video 

            min_thresh : The minimum threshold for setting the xlimits 

            outfile : Where to save the resulting video 

            MAX_FRAMES : The maximum number of frames to include per _FRAME_BREAK 

            _FRAME_BREAK : Only put _FRAME_BREAK seconds in each view 

            SLOWDOWN : Multiplier for slow down 

            figsz : Size of the figure/video

            mult : Multipler for how low to put text

    '''

    global ax, last, FRAME_BREAK, peaks_copy, mainline, ind, line, full_df, time_text

    # Copy over to global variables
    peaks_copy = peaks.copy()
    FRAME_BREAK = _FRAME_BREAK
    df = input_df.copy()
    full_df = df.copy()

    # The new rate is adjusted for the slowdown 
    new_rate = int(rate/float(SLOWDOWN))

    # Save the adio file 
    wav.write("audio/subset_sound.wav", new_rate, ssignal_df['signal'].values)

    # Total time of the video 
    TOTAL_TIME = float(df.index[-1] - df.index[0])

    # Maximum number of frames overal
    MAX_FRAMES = int(TOTAL_TIME / FRAME_BREAK * MAX_FRAMES)

    # Skip this many indices of the dataframe 
    SKIPS = df.shape[0] / MAX_FRAMES 
    
    # Shorten df 
    df = df[::SKIPS].copy()

    # Create figure and setup
    fig = plt.figure(figsize = figsz)
    ax = plt.axes(ylim = [np.min(np.min(input_df) * .8, 0), min_thresh*10.])

    # Main object for singal
    line, = ax.plot([], [], lw=2)

    # Line to say where we are in time
    mainline = ax.axvline(x = df.index[0], linewidth=2, c = 'k')

    # If there is a min threshold plot it
    if min_thresh:
        ax.axhline(y = min_thresh, linewidth=3, c = 'k')

    # Object to hold the text of what letter is being pressed
    time_text = plt.text(FRAME_BREAK / 2, min_thresh*5, '#START', fontsize=35)

    # Starting index 
    ind = 0

    # The start of the x axis 
    last = full_df.index[0] 

    # Function shifts the screen to next frame 
    def set_x(update = False):
        global last, full_df, peaks_copy, FRAME_BREAK, ax, time_text, ind
        
        # Update limits of view
        if update:
            last += FRAME_BREAK
        
        # Set new xlimit
        ax.set_xlim(last, last + FRAME_BREAK)

        # Update the text
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

    # Attempts to build the string to display on the screen
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
        
    #  Animate funciton for animation 
    def animate(i):
        global ind, line, last, mainline, FRAME_BREAK, ax, time_text

        # Update the time marker
        if mainline:
            mainline.set_xdata(df.index[i])
        else:
            mainline = ax.axvline(x = df.index[i], linewidth=1, c = 'k')
        
        # If we are ready for a new frame, set it
        if df.index[i] >= (last + FRAME_BREAK):
            line, = set_x(True)
        
        # If there are more peaks
        if ind < peaks_copy.shape[0]:
            
            # Determine the next peak we havent displayed
            next_peak =  peaks_copy.iloc[ind]['peak time']
            
            # For each peak that has been passed, display it
            while df.index[i] >= next_peak:

                # Update the text on the screen
                time_text.set_text(getstr(ind))

                # Draw a line to show where peak was
                if next_peak >= last:
                    ax.axvline(x = next_peak, linewidth=2, c = 'y')
                
                # Check to make sure next peak is not passed as well
                ind +=1
                if ind < peaks_copy.shape[0]:
                    next_peak =  peaks_copy.iloc[ind]['peak time']
                else:
                    next_peak =  1e15
        return line,

    # Calculate the number of FPS for the video
    fps = df.shape[0] / TOTAL_TIME / SLOWDOWN

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames= df.shape[0], interval=1, blit=True)

    # Write the animation to file
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'))#, bitrate=new_rate)
    anim.save('videos/im.mp4', writer=writer)
    
    # Use ffmpeg to merge audio and video
    subprocess.call(['ffmpeg', '-i', 'videos/im.mp4', '-i', 'audio/subset_sound.wav', '-c:v', 'copy', '-c:a',
                     'aac', '-strict', 'experimental', outfile, '-y'])

    # Filename for m4v file
    out = 'videos/' + os.path.splitext(outfile)[0] + '.m4v'

    # Convert to m4v
    subprocess.call(['ffmpeg', '-i', outfile, '-vcodec', 'libx264', out, '-y'])
    
    plt.close(fig)
    print('done')
    return out


def visualize_signal(signal_df, rate, outfile, audio_df, MAX_FRAMES = 20., _FRAME_BREAK = 5., 
                figsz = (12,8), mult = 2**9):
    '''
        Function to visualize clicks as an animated video created with matplotlib.animation 
        Combines the audio, with a video of the detected peaks 

            signal_df : Dataframe of raw signal during time period 

            rate : The rate to save the video 

            outfile : Where to save the resulting video 

            MAX_FRAMES : The maximum number of frames to include per _FRAME_BREAK 

            FRAME_BREAK : Only put _FRAME_BREAK seconds in each view 

            figsz : Size of the figure/video

            mult : Multipler for how low to put text

    '''
    global ax, last, FRAME_BREAK, mainline, ind, line, full_df
    FRAME_BREAK = _FRAME_BREAK
    basename = os.path.splitext(os.path.basename(outfile))[0]
    vid_dir = os.path.dirname(outfile)
    audio_fname = os.path.join(vid_dir, '..', 'audio', basename + '.wav')

    full_df = signal_df.copy()
    df = full_df.copy()

    # Save the audio file 
    wav.write(audio_fname, rate, audio_df['signal'].values)

    # Total time of the video 
    TOTAL_TIME = float(df.index[-1] - df.index[0])

    # Maximum number of frames overal
    MAX_FRAMES = (TOTAL_TIME * MAX_FRAMES) // FRAME_BREAK 

    # Skip this many indices of the dataframe 
    SKIPS = int(df.shape[0] // MAX_FRAMES )

    # SKIPS = 1
    print('TOTAL_TIME', TOTAL_TIME)
    print('SKIPS', SKIPS)
    print('MAX_FRAMES', MAX_FRAMES)
    print('df.shape[0]', df.shape[0])

    
    # Shorten df 
    df = df[::SKIPS].copy()

    # Create figure and setup
    fig = plt.figure(figsize = figsz)
    mx = np.max(np.max(df) * 1.05, 0)
    mn = np.min(np.min(df) * 1.05, 0)
    ax = plt.axes(ylim = [mn, mx])

    # Main object for singal
    line, = ax.plot([], [], lw=2)

    # Line to say where we are in time
    mainline = ax.axvline(x = df.index[0], linewidth=2, c = 'k')

    # Starting index 
    ind = 0

    # The start of the x axis 
    last = full_df.index[0] 

    # Function shifts the screen to next frame 
    def set_x(update = False):
        global last, full_df, FRAME_BREAK, ax, ind
        
        # Update limits of view
        if update:
            last += FRAME_BREAK
        
        # Set new xlimit
        ax.set_xlim(last, last + FRAME_BREAK)
        
        # Get dfs in view
        df_inds = ((full_df.index >= last) & (full_df.index <= (last + FRAME_BREAK)))
        sub_df = full_df[df_inds]
        
        # Plot the signals
        line, = ax.plot(sub_df.index.values, sub_df['signal'].values, 'b', lw=2)
        
        return line,

    # initialization function: plot the background of each frame
    def init():
        global ax
        line, = set_x()    
        return line,

   
    #  Animate funciton for animation 
    def animate(i):
        global ind, line, last, mainline, FRAME_BREAK, ax

        # Update the time marker
        if mainline:
            mainline.set_xdata(df.index[i])
        else:
            mainline = ax.axvline(x = df.index[i], linewidth=1, c = 'k')
        
        # If we are ready for a new frame, set it
        if df.index[i] >= (last + FRAME_BREAK):
            line, = set_x(True)
        
        return line,

    # Calculate the number of FPS for the video
    fps = df.shape[0] / TOTAL_TIME 

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames= df.shape[0], interval=1, blit=True)

    # Write the animation to file
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'))#, bitrate=new_rate)
    anim_loc = os.path.join(vid_dir, '%sanimation.mp4' % basename)
    anim.save(anim_loc, writer=writer)
    
    # Use ffmpeg to merge audio and video
    subprocess.call(['ffmpeg', '-i', anim_loc, '-i', audio_fname, '-c:v', 'copy', '-c:a',
                     'aac', '-strict', 'experimental', outfile, '-y'])

    # Filename for m4v file
    out = os.path.join(vid_dir, '%s.m4v' % basename)

    # Convert to m4v
    subprocess.call(['ffmpeg', '-i', outfile, '-vcodec', 'libx264', out, '-y'])
    
    plt.close(fig)
    os.remove(anim_loc)
    print('done')
    return out







def open_audio(raw_file, verbose = False, plt_every = 16, figsz = (12, 8)):
    '''
        Opens the audio file raw_file. If not .wav, this function will convert the file 
        to .wav before opening 

            raw_file : Name of the file 

            verbose : Whether or not to print 

            plt_every : When plotting raw signal, only plot every other plt_every

            figsz : The size of the figure 
    '''

    # Make sure file exists 
    if not os.path.exists(raw_file):
        print('file does not exist', raw_file)
        return None, None, None

    # Get the name
    filename, _ = os.path.splitext(raw_file)

    # Check if equivalent wav exists else make 
    wav_file = filename + '.wav'
    if not os.path.exists(wav_file):
        print('Converting %s to %s' % (raw_file, wav_file))
        subprocess.call(['ffmpeg', '-i', raw_file, wav_file])
    
    # Get the rate and raw signal
    (rate,sig) = wav.read(wav_file)

    # Only use one channel
    if len(sig.shape) > 1:
        sig = sig[:,0]

    # Save signal to a dataframe
    signal_df = pd.DataFrame(sig, columns = ['signal'])
    signal_df.index = signal_df.index.values / float(rate)
    signal_df.index.name = 'time (s)'

    # Scale signal to have standard deviation of 100
    signal_df['signal'] = signal_df['signal']  / signal_df['signal'].std()

    # Print out
    if verbose:
        print('.wav file location:', wav_file)
        print('rate:', rate, 'measurements per second')
        print('length of audio:', signal_df.index.max(), 'seconds')
        print('rate * length = ', signal_df.shape[0], 'measurements')
        signal_df['signal'][::plt_every].plot(title = 'Raw Measurements', figsize= figsz)

    return signal_df, rate, wav_file
