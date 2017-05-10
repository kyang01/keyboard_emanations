'''
    
'''

from .misc import *
from .FormWidgets import *

class Person(object):
	'''
		A representation of a person to share various recording of the 
		same person together for greater prediction power 
	'''

	def __init__(self, parent, name, decoder_nums = None):
		self.parent = parent
		self.processed = self.parent.processed
		self.decoder_folder = os.path.join(self.processed, 'decoders')
		if not os.path.exists(self.decoder_folder):
			os.mkdir(self.decoder_folder)

		self.name = name
		self.decoders = []

		if decoder_nums:
			self.add_decoders(decoder_nums)

	def add_decoders(self, decoder_nums):
		'''
			Adds a decoder to the current person
		'''
		# add to the list of decoders
		decoder_foldnums = filter(lambda x : x, decoder_nums.split(','))
		decoder_folders = map(lambda x : os.path.join(self.decoder_folder, x), decoder_foldnums)
		self.decoders = list(map(lambda x : Decoder(self, decode_folder = x), decoder_folders))

		# save to the process folder for later access
	
	def buildTree(self, branch):
		for decoder in self.decoders:
			# Make a branch for each view
			decoder.buildTree(branch)
			
	def get_random(self, key = None):
		'''
			Returns a random keystoke from this person, if key
			is not None, a random keystoke of character 'key'
			will be returned
		'''
		pass

	def confuse(self):
		'''
			Plays noises to confuse the detection algorithm 
		'''
		pass

class Decoder(object):
	'''
		Decoder object that will decode and predict text from
		an audio recording. If labels are attached, it will use
		them for validation purposes
	'''

	def __init__(self, parent, audio_file = None, input_text = None, output_text = None, actual_text = None, person = None, decode_folder = None):
		self.parent = parent
		self.processed = self.parent.processed
		self.decoder_folder = os.path.join(self.processed, 'decoders')
		if not os.path.exists(self.decoder_folder):
			os.mkdir(self.decoder_folder)
		self.vid_types = {
			'ahistogram' : "[0:a]ahistogram,format=yuv420p[v]",
			'showwaves' : "[0:a]showwaves=s=1280x720:mode=line:rate=25,format=yuv420p[v]",
			'showcqt' : "[0:a]showcqt,format=yuv420p[v]",
			'showfreqs' : "[0:a]showfreqs=mode=line:fscale=log,format=yuv420p[v]",
			'showspectrum' : "[0:a]showspectrum=s=1280x720,format=yuv420p[v]",
		}
	
		
		# Holds the various audio representations in memory

		self.audios = {}

		if decode_folder:
			self.signal_directory = os.path.join(decode_folder, 'signals')
	
			self.video_directory = os.path.join(decode_folder, 'video')
		
			self.audio_directory = os.path.join(decode_folder, 'audio')
			self.threshold_directory = os.path.join(decode_folder, 'threshold')
	

			self.init_folder(decode_folder)

			
		else:
			self.decode_folder = None

			# the audio file we want to decode
			self.audio_file = audio_file

			# OPTIONAL: the actual text that was written
			self.input_text = input_text

			# OPTIONAL: the labeled characters and when they were typed
			self.output_text = output_text

			# OPTIONAL: if we were typing a specific piece of text, what is it
			self.actual_text = actual_text

			# OPTIONAL: is this recording linked to a specific person that we have
			# 			attempted to decode before
			self.person = person

			# Extract whatever metadata possible from the filenames
			self.metadata = self.extract_metadata(self.audio_file)

			# The location of video files for various of audio files
			self.videos = {}
					
	def init_folder(self, folder):
		self.decode_folder = folder
		self.signal_files = glob.glob(os.path.join(self.signal_directory, '*.csv'))
		self.keys = list(map(lambda x : os.path.splitext(os.path.basename(x))[0], self.signal_files))

		# the audio file we want to decode
		flatten = lambda l: [item for sublist in l for item in sublist]

		self.videos = {}
		self.audios = {}
		self.signals = {}
		for key in self.keys:
			# The location of video files for various of files
			self.videos[key] = glob.glob(os.path.join(self.video_directory, '%s*.mp4' % key))
			
			self.signals[key] = glob.glob(os.path.join(self.signal_directory, '%s*.mp4' % key))

			all_audio = flatten([glob.glob(os.path.join(self.audio_directory, key + ext)) for ext in config.AUDIO_EXTS])
			if len(all_audio) > 0:
				self.audios[key] = [all_audio[0]]
			else:
				self.audios[key] = []


		# OPTIONAL: the actual text that was written
		self.input_text = os.path.join(folder, 'input_text.txt')

		# OPTIONAL: the labeled characters and when they were typed
		self.output_text = os.path.join(folder, 'output_text.txt')

		# OPTIONAL: if we were typing a specific piece of text, what is it
		self.actual_text = os.path.join(folder, 'actual_text.txt')

		# Extract whatever metadata possible from the filenames
		self.metadatafname= os.path.join(folder, 'metadata.csv')
		self.metadata = self.extract_metadata_from_file(self.metadatafname)
		self.person = self.metadata['person']

	def extract_metadata_from_file(self, fname):
		'''
		'''
		return pd.Series.from_csv(fname).to_dict()

	def add_metadata_item(self, key, value):
		metadata = pd.Series.from_csv(self.metadatafname)
		metadata.ix[key] = value
		metadata.to_csv(os.path.join(self.decode_folder, 'metadata.csv'))

	def add_metadata_item_callback(self):
		self.w = MetadataDisplay(self)
		self.w.show()

	def rebuild(self):
		'''
		'''
		self.parent.parent.buildTree()

	def getname(self):
		'''
		'''
		return self.metadata['directory-name']

	@classmethod
	def extract_metadata(self, fname):
		'''
			extracts metadata from file names for later analyeses for outside
		'''
		directory = os.path.dirname(fname)
		dir_name = os.path.basename(directory)
		metadata = {'directory' : directory, 
				'fname' : os.path.basename(fname),
				'directory-name' : dir_name,
				'person' : None
				}

		dir_name_splt = dir_name.split('_')
		if len(dir_name_splt) == 6:
			metadata['actual-text'] = dir_name_splt[0]
			metadata['shift'] = dir_name_splt[1]
			metadata['punctuation'] = dir_name_splt[2]
			metadata['background-noise'] = dir_name_splt[3]
			metadata['speed'] = dir_name_splt[4]
			metadata['recording-device'] = dir_name_splt[5]

		return metadata

	def update_display(self, display_area):
		
		return DecoderDisplay(self)

	def extract_metadata_internal(self):
		'''
			extracts metadata from file names for later analyeses
		'''
		metadata = self.extract_metadata(self.audio_file)
		metadata['person'] = self.person
		return metadata

	def buildTree(self, root):
		branch = DecoderTreeWidget(root, [self.getname()], self)

		for audio_key in self.keys:
			new_branch = AudioTreeWidget(branch, [audio_key], self.audios[audio_key], self.videos[audio_key])

	def save(self):
		'''
			Save the decoder object to the processing folder
		'''
		# Get list of decoders
		decoders, _ = split_directory(self.decoder_folder)

		# old folders
		sorted_decoder_labels = sorted(list(map(int, decoders)))

		fold_num = 0
		if len(sorted_decoder_labels) > 0:
			fold_num = sorted_decoder_labels[-1] + 1

		# the new folder for this decoder
		new_folder = os.path.join(self.decoder_folder, str(fold_num))
		if os.path.exists(new_folder):
			assert(False) #error
		os.mkdir(new_folder)

		self.signal_directory = os.path.join(new_folder, 'signals')
		os.mkdir(self.signal_directory)
		self.video_directory = os.path.join(new_folder, 'video')
		os.mkdir(self.video_directory)
		self.audio_directory = os.path.join(new_folder, 'audio')
		os.mkdir(self.audio_directory)
		self.threshold_directory = os.path.join(new_folder, 'threshold')
		os.mkdir(self.threshold_directory)


		# copy the audio file
		audio_ext = os.path.splitext(self.audio_file)[1]
		shutil.copy(self.audio_file, os.path.join(self.audio_directory, 'raw' + audio_ext))

		# open and save the signal to file
		signal_fname = os.path.join(self.signal_directory, 'raw.csv')
		signal_df, rate, _ = spl.open_audio(self.audio_file, verbose = True, plt_every = 2**8)
		signal_df.to_csv(signal_fname)


		# copy the input_text file
		if self.input_text:
			shutil.copy(self.input_text, os.path.join(new_folder, 'input_text.txt'))

		# copy the output_text file
		if self.output_text:
			shutil.copy(self.output_text, os.path.join(new_folder, 'output_text.txt'))

		# copy the actual_text file
		if self.actual_text:
			shutil.copy(self.actual_text, os.path.join(new_folder, 'actual_text.txt'))
		
		#save metadata
		metadata = pd.Series(self.extract_metadata_internal())
		metadata['raw_rate'] = rate
		metadata.to_csv(os.path.join(new_folder, 'metadata.csv'))

		# add to the persons df the existing decoder
		self.parent.update_person(self.person, fold_num)

	def assign_person(self):
		'''
			Assigns decoder to a specific person
		'''
		self.person = person

	def visualize_signal(self):
		mapp = self.parent.parent.parent.parent
		# tree item 
		item = self.parent.parent.tree.currentItem()
		if type(item) != AudioTreeWidget:
			QMessageBox.warning(mapp, 'Could not visualize signal!',
											"Select an audio signal, try raw" ,
											QMessageBox.Ok)
			return

		self.w = VisSignalDisplay(self, item, mapp)
		self.w.show()
		
	def visualize_done(self):
		mapp = self.parent.parent.parent.parent
		mapp.progress_bar.setValue(mapp.progress_bar.value() + 1)

	def vis_fin(self):
		mapp = self.parent.parent.parent.parent

		QMessageBox.information(mapp, 'Finished Converting Videos!',
										"Finished Creating visualization of signal" ,
										QMessageBox.Ok)
		self.parent.parent.buildTree()
		mapp.resetLabel()
			
	def create_video(self):
		'''
			Creates a video of an audio recording
		'''
		mapp = self.parent.parent.parent.parent
		# tree item 
		item = self.parent.parent.tree.currentItem()
		if type(item) != AudioTreeWidget:
			QMessageBox.warning(mapp, 'Could not create video!',
											"Select the audio type to create video, try raw" ,
											QMessageBox.Ok)
			return

		if len(item.audio_files) == 0:
			QMessageBox.warning(mapp, 'Could not create video!',
											"No audio recording for selected, try raw" ,
											QMessageBox.Ok)
			return
		
		self.w = VideoDisplay(self, item, mapp, self.vid_types)
		self.w.show()
			
	def done(self):
		mapp = self.parent.parent.parent.parent
		mapp.progress_bar.setValue(mapp.progress_bar.value() + 1)

	def fini(self):
		mapp = self.parent.parent.parent.parent
		QMessageBox.information(mapp, 'Finished Converting Videos!',
										"Finished Converting Videos" ,
										QMessageBox.Ok)
		self.parent.parent.buildTree()
		mapp.resetLabel()

	def finished_thresholding(self):
		mapp = self.parent.parent.parent.parent

		QMessageBox.information(mapp, 'Finished Thresholding!',
										"Finished Thresholding" ,
										QMessageBox.Ok)

		fname = os.path.join(self.threshold_directory, 'FigureObject.peaks.pickle')
		figx = pickle.load(open(fname, 'rb'))

		figx.show() # Show the figure, edit it, etc.!
		self.parent.parent.buildTree()
		mapp.resetLabel()

	def done_transform_audio(self):
		mapp = self.parent.parent.parent.parent
		mapp.progress_bar.setValue(mapp.progress_bar.value() + 1)

	def finished_transform_audio(self):
		mapp = self.parent.parent.parent.parent
		self.parent.parent.buildTree()

		QMessageBox.information(mapp, 'Finished Transforming Audio!',
										"Finished Transforming Audio" ,
										QMessageBox.Ok)
		mapp.resetLabel()

	def transform_audio(self, VERBOSE = True):
		'''
			Transforms the raw audio into another representation
		'''
		
		mapp = self.parent.parent.parent.parent
		self.w = TransAudioDisplay(self, mapp)
		self.w.show()
	
	def threshold_keystrokes(self, threshold):
		'''
			Thresholds the audio 
		'''
		mapp = self.parent.parent.parent.parent
		item = self.parent.parent.tree.currentItem()
		audio_type = item.values[0]
		if audio_type != 'fourier':
			QMessageBox.warning(mapp, 'Could not visualize signal!',
											"Currently only fourier" ,
											QMessageBox.Ok)
			return

		
		self.w = ThresholdDisplay(self, mapp, audio_type)
		self.w.show()

	def finished_clustering(self):
		mapp = self.parent.parent.parent.parent

		QMessageBox.information(mapp, 'Finished Clustering!',
										"Finished Clustering" ,
										QMessageBox.Ok)


		self.parent.parent.buildTree()
		mapp.resetLabel()

	def cluster_keystrokes(self):
		'''
			K-means cluster the currently created keystrokes 
		'''
		
		mapp = self.parent.parent.parent.parent
		self.w = ClusterDisplay(self, mapp)
		self.w.show()
		
	def predict_text(self):
		'''
			Run an hmm of the current clustered keystokes to predict text
		'''
		
		mapp = self.parent.parent.parent.parent
		self.w = PredictDisplay(self, mapp)
		self.w.show()
		
	def confuse_attacker(self):
		'''
			Run an hmm of the current clustered keystokes to predict text
		'''
		
		 
		mapp = self.parent.parent.parent.parent
		self.w = ConfuseDisplay(self, mapp)
		self.w.show()
		
	def get_random_keystoke(self, key = None):
		'''
			Returns the sound of a random keystroke, if key is None
			it will return a random sound, if key is a specific character,
			then a random keystroke of that character will be returned if it exists
		'''
		#TODO 
		pass