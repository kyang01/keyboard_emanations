'''
    
'''

from .misc import *
from .defense import DefenseBackgroundThread



class BackgroundThread(QThread):
	'''
		Generic background thread
	'''
	add_post = pyqtSignal(name='add_post')

	def __init__(self, parent, done, finished, inputs):
		super(BackgroundThread, self).__init__(parent)
		self.parent = parent
		self.inputs = inputs

		# Set up connection to send signal that updates progress bar
		self.add_post.connect(done)
		self.finished.connect(finished)

class CreateDecoderBackgroundThread(BackgroundThread):
	def __init__(self, parent, done, finished, inputs):
		super(CreateDecoderBackgroundThread, self).__init__(parent,  done, finished, inputs)

	def run(self):
		'''
	
		'''
		self.add_post.emit()
		self.inputs['decoder'](self.inputs['parent'], mapp = self.inputs['mapp'], 
							audio_file = self.inputs['current_file'], 
                            input_text = self.inputs['input_text'], 
                            output_text = self.inputs['output_text'], 
                            actual_text = self.inputs['actual_text'], 
                            person = self.inputs['person']).save(signal = self.add_post)
		self.add_post.emit()

class VisualizeSignalBackgroundThread(BackgroundThread):
	'''
	'''

	def __init__(self, parent, done, finished, inputs):
		super(VisualizeSignalBackgroundThread, self).__init__(parent,  done, finished, inputs)
		
	def run(self):
		'''
	
		'''
		self.add_post.emit()
		spl.visualize_signal(signal_df = self.inputs['signal_df'], rate = self.inputs['rate'],
							outfile = self.inputs['outfile'], audio_df = self.inputs['audio_df'], 
							_FRAME_BREAK = self.inputs['_FRAME_BREAK'], MAX_FRAMES = self.inputs['MAX_FRAMES'], 
							signal = self.add_post)
		self.add_post.emit()

class ConvertVideoBackgroundThread(BackgroundThread):
	'''
	'''

	def __init__(self, parent, done, finished, inputs):
		super(ConvertVideoBackgroundThread, self).__init__(parent, done, finished, inputs)

	def run(self):
		'''
		'''
		for vid_type in tqdm(self.inputs['vid_types'].keys()):
			directory = os.path.dirname(self.inputs['input_file'])
			output_file = os.path.join(directory, '..', 'video', '%s_%s.mp4' % (self.inputs['audio_type'], vid_type))
			ff = ffmpy.FFmpeg(
					inputs={self.inputs['input_file']: None},
					outputs={output_file: '-y -filter_complex "%s" -map "[v]" -map 0:a' % (self.inputs['vid_types'][vid_type])})
			ff.run()
			self.add_post.emit()

class TransformAudioBackgroundThread(BackgroundThread):
	'''
	'''

	def __init__(self, parent, done, finished, inputs):
		super(TransformAudioBackgroundThread, self).__init__(parent, done, finished, inputs)
		self.VERBOSE = True

	def run(self):
		'''
			
		'''
		signal_df, rate, wav_file = spl.open_audio(self.inputs['audio_file'], 
												verbose = self.VERBOSE, plt_every = 2**8)
		self.add_post.emit()
		freqs, fourier_df = spl.get_windowed_fourier(signal_df, rate, 
											MIN_FREQ = self.inputs['MIN_FREQ'], 
											 MAX_FREQ = self.inputs['MAX_FREQ'], 
											 verbose = self.VERBOSE)
		self.inputs['parent'].add_metadata_item('max-fourier-signal', fourier_df['signal'].max())
		self.add_post.emit()
		fourier_df.to_csv(self.inputs['output_loc'])

class ThresholdBackgroundThread(BackgroundThread):
	'''
		A generic background thread for our application 
	'''
	add_post = pyqtSignal(name='add_post')

	def __init__(self, parent, done, finished, inputs):
		super(ThresholdBackgroundThread, self).__init__(parent, done, finished, inputs)

	def run(self):
		'''
		'''
		fourier_df = pd.read_csv(self.inputs['fourier_floc']).set_index('time (s)')
		signal_df = pd.read_csv(self.inputs['signal_floc']).set_index('time (s)')

		if os.path.exists(self.inputs['output_text']):
			true_df = spl.clean_output_text(self.inputs['output_text'])
		else:
			true_df = None

		sfourier_df, ssignal_df, all_peaks = spl.detect_peaks(fourier_df, signal_df, true_df = true_df,
                                              t0 = self.inputs['MINT'], t1 = self.inputs['MAXT'] , 
                                              min_thresh = self.inputs['MIN_THRESH'],
                                              max_thresh = self.inputs['MAX_THRESH'],
                                               min_dist = self.inputs['MIN_DIST'], 
                                              key_len = self.inputs['KEY_LEN'], 
                                              back_prop = self.inputs['BACK_PROP'], 
                                              save_dir = self.inputs['save_dir'],
                                              to_add = self.inputs['TO_ADD'], signal = self.add_post)

		signal_save = os.path.join(self.inputs['save_dir'], 'raw.csv')
		ssignal_df.to_csv(signal_save)
		self.add_post.emit()
		fourier_save = os.path.join(self.inputs['save_dir'], 'fourier.csv')
		sfourier_df.to_csv(fourier_save)
		self.add_post.emit()

		peaks_save = os.path.join(self.inputs['save_dir'], 'peaks.csv')
		all_peaks.to_csv(peaks_save)
		self.add_post.emit()

		ML_INPUT = spl.build_input_df(ssignal_df, all_peaks)
		self.add_post.emit()
		ml_save = os.path.join(self.inputs['save_dir'], 'processing_input.csv')
		ML_INPUT.to_csv(ml_save)

class ClusterBackgroundThread(BackgroundThread):
	'''
		A generic background thread for our application 
	'''

	def __init__(self, parent, done, finished, inputs):
		super(ClusterBackgroundThread, self).__init__(parent, done, finished, inputs)

	def run(self):
		'''
			Changes the status bar to say the parent tool is running 
			and then calls the process function
		'''
		self.add_post.emit()
		INPUT_FILE = os.path.join(self.inputs['save_dir'], 'processing_input.csv')
		char_inps = pd.read_csv(INPUT_FILE, index_col=0)
		self.add_post.emit()

		cepstrum_df = pl.extract_cepstrum(char_inps, self.inputs['rate'], 
									mfcc_start=self.inputs['MFCC_START'], 
									mfcc_end=self.inputs['MFCC_END'],
									winlen = self.inputs['winlen'], 
									winstep = self.inputs['winstep'],
            					numcep = self.inputs['numcep'], 
            					nfft= self.inputs['nfft'],
            					nfilt = self.inputs['nfilt'], 
            					lowfreq = self.inputs['lowfreq'], 
            					highfreq = self.inputs['highfreq'])
		self.add_post.emit()

		cepstrum_df = pl.cluster(cepstrum_df, num_clusters = self.inputs['NUM_CLUSTERS'], 
										n_components = self.inputs['N_COMPONENTS'])
		self.add_post.emit()

		OUTPUT_FILE = os.path.join(self.inputs['save_dir'], 'processing_clusters.csv')
		cepstrum_df.to_csv(OUTPUT_FILE)

class PredictBackgroundThread(BackgroundThread):
	'''
	'''

	def __init__(self, parent, done, finished, inputs):
		super(PredictBackgroundThread, self).__init__(parent, done, finished, inputs)
		

	def run(self):
		'''
			Changes the status bar to say the parent tool is running 
			and then calls the process function
		'''
		self.add_post.emit()
		print('predict backg')