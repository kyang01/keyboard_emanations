'''
    
'''

from .misc import *

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
	send_estimate = pyqtSignal(str, name='send_estimate')

	def __init__(self, parent, done, finished, inputs):
		super(PredictBackgroundThread, self).__init__(parent, done, finished, inputs)
		self.send_estimate.connect(inputs['display_prediction'])
		
	def run(self):
		'''
			Changes the status bar to say the parent tool is running 
			and then calls the process function
		'''
		self.add_post.emit()


		cepstrum_df = pd.read_csv(self.inputs['cepstrum_df_floc'], index_col=0)
		self.add_post.emit()

		A_df, n_unique, unique_chars, id_to_char, char_to_id  = pl.build_transmission_full()
		self.add_post.emit()


		#Build emissions matrix
		Eta = pl.build_eta(cepstrum_df, unique_chars, self.inputs['NUM_CLUSTERS'],
												 do_all = self.inputs['DO_ALL'])
		self.add_post.emit()

		# runn hmm model
		estimate, acc, acc_wospace, score, hmm = pl.run_hmm(cepstrum_df, self.inputs['targ_s'], 
		                                                    self.inputs['NUM_CLUSTERS'], t_smooth = self.inputs['smooth'], 
		                                                    tol = self.inputs['TOL'],
		                                                    do_all = self.inputs['DO_ALL'], verbose = self.inputs['VERBOSE'])  

		self.send_estimate.emit(estimate)

		self.add_post.emit()

class DefenseBackgroundThread(BackgroundThread):
    '''
        Plays keyboard sounds to interfere with the detection algorithm.
        Start and End are the only important public API functions
    '''
    def __init__(self, parent, done, finished, inputs):
        super(DefenseBackgroundThread, self).__init__(parent, done, finished, inputs)
        self.defending = False
        self.interfering = False
        
        # Set up sound files
        key_sound = sa.WaveObject.from_wave_file("app/core/assets/KeyPress.wav")
        space_sound = sa.WaveObject.from_wave_file("app/core/assets/SpacePress.wav")
        multi_sound = sa.WaveObject.from_wave_file("app/core/assets/FastKeys.wav")
        self.sounds = [key_sound, space_sound, multi_sound]

    def run(self):
        self.startDefense()
    
    def startDefense(self):
        print("start")
        self.defending = True
        # Listen to keyboard
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def endDefense(self):
        print("stop")
        self.defending = False
        keyboard.Listener.StopException

    # Only detects special keys (shift, option...) due to OS X security
    def on_press(self, key):
        if not self.interfering:
            self.playInterference()

    def playInterference(self):
        self.interfering = True
        play_at = random.exponential(0.1)
        
        # print("play at " + str(play_at))
        
        # Wait, then play sound
        time.sleep(play_at)
        
        # Play sound
        wav_obj = random.choice(self.sounds, p=[0.4,0.4,0.2])
        play_obj = wav_obj.play()
        play_obj.wait_done()
        
        # 60% possibility of recurrance
        if random.rand() > 0.4:
            # Continue keystrokes recursively
            self.playInterference()
        else:
            self.interfering = False
            