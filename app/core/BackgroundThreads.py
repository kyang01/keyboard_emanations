'''
    
'''

from .misc import *
from .defense import *

class VisualizeSignalBackgroundThread(QThread):
	'''
		A generic background thread for our application 
	'''
	add_post = pyqtSignal(name='add_post')

	def __init__(self, parent, signal_df, rate, outfile, audio_df, MAX_FRAMES, _FRAME_BREAK, START_TIME, END_TIME, done, finished):
		super(VisualizeSignalBackgroundThread, self).__init__(parent)
		self.parent = parent
		self.signal_df = signal_df
		self.rate = rate
		self.outfile = outfile
		self.MAX_FRAMES = MAX_FRAMES
		self._FRAME_BREAK = _FRAME_BREAK
		self.START_TIME = START_TIME
		self.END_TIME = END_TIME
		self.audio_df = audio_df

		# Set up connection to send signal that updates progress bar
		self.add_post.connect(done)
		self.finished.connect(finished)

	def run(self):
		'''
			Changes the status bar to say the parent tool is running 
			and then calls the process function
		'''

		#TODO 
		self.add_post.emit()

		spl.visualize_signal(signal_df = self.signal_df, rate = self.rate,
							outfile = self.outfile, audio_df = self.audio_df, 
							_FRAME_BREAK = self._FRAME_BREAK, MAX_FRAMES = self.MAX_FRAMES, signal = self.add_post)

		self.add_post.emit()
		

class TransformAudioBackgroundThread(QThread):
	'''
		A generic background thread for our application 
	'''
	add_post = pyqtSignal(name='add_post')

	def __init__(self, parent, audio_file, output_loc, MIN_FREQ, MAX_FREQ, done, finished):
		super(TransformAudioBackgroundThread, self).__init__(parent)
		self.parent = parent
		self.audio_file = audio_file
		self.output_loc = output_loc
		self.MIN_FREQ = MIN_FREQ
		self.MAX_FREQ = MAX_FREQ
		self.VERBOSE = True

		# Set up connection to send signal that updates progress bar
		self.add_post.connect(done)
		self.finished.connect(finished)

	def run(self):
		'''
			Changes the status bar to say the parent tool is running 
			and then calls the process function
		'''

		#TODO 
		signal_df, rate, wav_file = spl.open_audio(self.audio_file, verbose = self.VERBOSE, plt_every = 2**8)
		self.add_post.emit()
		freqs, fourier_df = spl.get_windowed_fourier(signal_df, rate, MIN_FREQ = self.MIN_FREQ, 
											 MAX_FREQ = self.MAX_FREQ, verbose = self.VERBOSE)
		self.add_post.emit()
		fourier_df.to_csv(self.output_loc)

class ConvertVideoBackgroundThread(QThread):
	'''
		A generic background thread for our application 
	'''
	add_post = pyqtSignal(name='add_post')

	def __init__(self, parent, vid_types, input_file, audio_type,  done, finished):
		super(ConvertVideoBackgroundThread, self).__init__(parent)
		self.parent = parent
		self.vid_types = vid_types
		self.input_file = input_file
		self.audio_type = audio_type
		# Set up connection to send signal that updates progress bar
		self.add_post.connect(done)
		self.finished.connect(finished)

	def run(self):
		'''
			Changes the status bar to say the parent tool is running 
			and then calls the process function
		'''


		for vid_type in tqdm(self.vid_types.keys()):


			directory = os.path.dirname(self.input_file)
			output_file = os.path.join(directory, '..', 'video', '%s_%s.mp4' % (self.audio_type, vid_type))
			ff = ffmpy.FFmpeg(
					inputs={self.input_file: None},
					outputs={output_file: '-y -filter_complex "%s" -map "[v]" -map 0:a' % (self.vid_types[vid_type])})
			ff.run()
			self.add_post.emit()

class ThresholdBackgroundThread(QThread):
	'''
		A generic background thread for our application 
	'''
	add_post = pyqtSignal(name='add_post')

	def __init__(self, parent, MINT, MAXT, MIN_THRESH, MAX_THRESH, MIN_DIST, 
							KEY_LEN, BACK_PROP, TO_ADD, fourier_floc, signal_floc, save_dir, 
							output_text, done, finished):
		super(ThresholdBackgroundThread, self).__init__(parent)
		self.parent = parent
		self.MINT = MINT
		self.MAXT = MAXT
		self.MIN_THRESH = MIN_THRESH
		self.MAX_THRESH = MAX_THRESH
		self.MIN_DIST = MIN_DIST
		self.KEY_LEN = KEY_LEN
		self.BACK_PROP = BACK_PROP
		self.fourier_floc = fourier_floc 
		self.signal_floc = signal_floc 
		self.save_dir = save_dir
		self.output_text = output_text
		self.TO_ADD = TO_ADD


		# Set up connection to send signal that updates progress bar
		self.add_post.connect(done)
		self.finished.connect(finished)

	def run(self):
		'''
			Changes the status bar to say the parent tool is running 
			and then calls the process function
		'''


		fourier_df = pd.read_csv(self.fourier_floc).set_index('time (s)')
		
		signal_df = pd.read_csv(self.signal_floc).set_index('time (s)')

		if os.path.exists(self.output_text):
			true_df = spl.clean_output_text(self.output_text)
		else:
			true_df = None

		sfourier_df, ssignal_df, all_peaks = spl.detect_peaks(fourier_df, signal_df, true_df = true_df,
                                              t0 = self.MINT, t1 = self.MAXT , min_thresh = self.MIN_THRESH,
                                              max_thresh = self.MAX_THRESH, min_dist = self.MIN_DIST, 
                                              key_len = self.KEY_LEN, back_prop = self.BACK_PROP, save_dir = self.save_dir,
                                              to_add = self.TO_ADD, signal = self.add_post)

		signal_save = os.path.join(self.save_dir, 'raw.csv')
		ssignal_df.to_csv(signal_save)
		self.add_post.emit()
		fourier_save = os.path.join(self.save_dir, 'fourier.csv')
		sfourier_df.to_csv(fourier_save)
		self.add_post.emit()

		peaks_save = os.path.join(self.save_dir, 'peaks.csv')
		all_peaks.to_csv(peaks_save)
		self.add_post.emit()

		ML_INPUT = spl.build_input_df(ssignal_df, all_peaks)
		ml_save = os.path.join(self.save_dir, 'processing_input.csv')
		ML_INPUT.to_csv(ml_save)

class ClusterBackgroundThread(QThread):
	'''
		A generic background thread for our application 
	'''
	add_post = pyqtSignal(name='add_post')

	def __init__(self, parent, save_dir, rate, done, finished, MFCC_START = 0, MFCC_END = -1,
				winlen = 0.01, winstep = 0.0025, numcep = 16, nfilt = 32, lowfreq = 400,
				highfreq = 12000, NUM_CLUSTERS = 40, N_COMPONENTS = 100):
		super(ClusterBackgroundThread, self).__init__(parent)
		self.parent = parent
		self.save_dir = save_dir
		self.rate = rate
		self.MFCC_START = MFCC_START
		self.MFCC_END = MFCC_END
		self.winlen = winlen
		self.winstep = winstep
		self.numcep = numcep
		self.nfilt = nfilt
		self.lowfreq = lowfreq
		self.highfreq = highfreq
		self.NUM_CLUSTERS = NUM_CLUSTERS
		self.N_COMPONENTS  = N_COMPONENTS

		# Set up connection to send signal that updates progress bar
		self.add_post.connect(done)
		self.finished.connect(finished)

	def run(self):
		'''
			Changes the status bar to say the parent tool is running 
			and then calls the process function
		'''
		self.add_post.emit()
		INPUT_FILE = os.path.join(self.save_dir, 'processing_input.csv')
		char_inps = pd.read_csv(INPUT_FILE, index_col=0)
		self.add_post.emit()

		cepstrum_df = pl.extract_cepstrum(char_inps, self.rate, 
									mfcc_start=self.MFCC_START, mfcc_end=self.MFCC_END,
									winlen = self.winlen, winstep = self.winstep,
            					numcep = self.numcep, nfilt = self.nfilt, 
            					lowfreq = self.lowfreq, highfreq = self.highfreq)
		self.add_post.emit()

		cepstrum_df = pl.cluster(cepstrum_df, num_clusters = self.NUM_CLUSTERS, n_components = self.N_COMPONENTS)
		self.add_post.emit()

		OUTPUT_FILE = os.path.join(self.save_dir, 'processing_clusters.csv')
		cepstrum_df.to_csv(OUTPUT_FILE)

