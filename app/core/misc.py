# GUI tools
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *



# File systems
import os, sys, glob, time, shutil
STDOUT = sys.stdout
import subprocess
from functools import partial
import json
import itertools
import textwrap
from tqdm import tqdm

# Configuration parameters
from app import config

import pandas as pd
import ffmpy

sys.path.append('Libraries')
import signal_processing as spl 

def split_directory(directory):
	'''
		Returns dictionary of folders and files 
		in a directory 
	'''
	folds = {}
	fils = {}

	for fold in glob.glob(os.path.join(str(directory), '*')):
		if '.' not in fold:
			folds[os.path.basename(fold)] = fold
		else:
			fils['.'.join(fold.split('/')[-1].split('.')[:-1])] = fold

	return folds, fils


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
		return pd.Series.from_csv(fname).to_dict()

	def add_metadata_item(self, key, value):
		metadata = pd.Series.from_csv(self.metadatafname)
		metadata.ix[key] = value
		metadata.to_csv(os.path.join(self.decode_folder, 'metadata.csv'))




	def getname(self):
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
		class DecoderDisplay(QWidget):
			def __init__(self, parent):
				QWidget.__init__(self)
				self.parent = parent
				self.buildUI()

			def buildUI(self):
				# We use the grid layout to create the ui
				grid = QGridLayout()
				grid.setSpacing(10)

				grid.addWidget(QLabel("KEY:"), 0, 1)
				grid.addWidget(QLabel('VALUE'), 0, 2)
				x = 1
				for key in self.parent.metadata.keys():
					grid.addWidget(QLabel(key + ":"), x, 1)
					grid.addWidget(QLabel(self.parent.metadata[key]), x, 2)
					x += 1

				# transform audio using cepstrum
				visualize_signal = QPushButton("Visualize Signal")
				visualize_signal.clicked.connect(self.parent.visualize_signal)
				grid.addWidget(visualize_signal, 0, 0)

				# create a video of the currently selected audio
				create_video = QPushButton("Create Video")
				create_video.clicked.connect(self.parent.create_video)
				grid.addWidget(create_video, 1, 0)

				# transform audio using cepstrum
				transform_audio = QPushButton("Transform Audio")
				transform_audio.clicked.connect(self.parent.transform_audio)
				grid.addWidget(transform_audio, 2, 0)

				# thresold into keystrokes
				threshold_keystrokes = QPushButton("Threshold Keystrokes")
				threshold_keystrokes.clicked.connect(self.parent.threshold_keystrokes)
				grid.addWidget(threshold_keystrokes, 3, 0)

				# cluster keystrokes using kmeans
				cluster_keystrokes = QPushButton("Cluster Keystrokes")
				cluster_keystrokes.clicked.connect(self.parent.cluster_keystrokes)
				grid.addWidget(cluster_keystrokes, 4, 0)

				# predict the text of the keystrokes
				predict_text = QPushButton("Predict Text")
				predict_text.clicked.connect(self.parent.predict_text)
				grid.addWidget(predict_text, 5, 0)

				self.setLayout(grid)

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
		

		audio_type = item.values[0]

		signal_floc = os.path.join(self.signal_directory, '%s.csv' % audio_type)
		signal_df = pd.read_csv(signal_floc).set_index('time (s)')

		output_loc = os.path.join(self.video_directory, '%s_signal.mp4' % audio_type)

		mapp = self.parent.parent.parent.parent
		mapp.progress_bar.setMaximum(2)
		mapp.status_label.setText('Visualizing signal...')

		if audio_type == 'raw':
			audio_df = signal_df
		else:
			audio_floc = os.path.join(self.signal_directory, 'raw.csv')
			audio_df = pd.read_csv(audio_floc).set_index('time (s)')

		
		bk_thrd = VisualizeSignalBackgroundThread(parent = self.parent.parent,
													signal_df = signal_df, 
													rate = int(self.metadata['%s_rate' % audio_type]),
													outfile = output_loc,
													audio_df =  audio_df,
													done = self.visualize_done)
		bk_thrd.start()

	def visualize_done(self):
		mapp = self.parent.parent.parent.parent
		mapp.progress_bar.setValue(mapp.progress_bar.value() + 1)

		if mapp.progress_bar.value() == 2:
			self.finished_transform_audio()

			


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
		input_file = item.audio_files[0]
		audio_type = item.values[0]

		
		mapp = self.parent.parent.parent.parent
		mapp.progress_bar.setMaximum(len(self.vid_types))
		mapp.status_label.setText('Converting Audio files to video representation...')

		
		
		bk_thrd = ConvertVideoBackgroundThread(self.parent.parent, self.vid_types, input_file, audio_type, self.done)
		bk_thrd.start()
			
	def done(self):
		mapp = self.parent.parent.parent.parent
		mapp.progress_bar.setValue(mapp.progress_bar.value() + 1)

		if mapp.progress_bar.value() == len(self.vid_types):
			self.parent.parent.buildTree()

	
			QMessageBox.information(mapp, 'Finished Converting Videos!',
                                            "Finished Converting Videos" ,
                                            QMessageBox.Ok)
			mapp.resetLabel()


	def done_transform_audio(self):
		mapp = self.parent.parent.parent.parent
		mapp.progress_bar.setValue(mapp.progress_bar.value() + 1)

		if mapp.progress_bar.value() == 3:
			self.finished_transform_audio()

	def finished_transform_audio(self):
		mapp = self.parent.parent.parent.parent
		self.parent.parent.buildTree()


		QMessageBox.information(mapp, 'Finished Converting Videos!',
                                        "Finished Converting Videos" ,
                                        QMessageBox.Ok)
		mapp.resetLabel()

	def transform_audio(self, VERBOSE = True):
		'''
			Transforms the raw audio into another representation
		'''

		mapp = self.parent.parent.parent.parent
		mapp.progress_bar.setMaximum(3)
		mapp.status_label.setText('Transforming Audio Files...')


		# The minimum frequency to keep for fourier
		MIN_FREQ = 400

		# The maximum frequency to keep for fourier
		MAX_FREQ = 12000

		output_loc = os.path.join(self.signal_directory, 'cepstrum.csv')
		
		bk_thrd = TransformAudioBackgroundThread(parent = self.parent.parent,
											audio_file = self.audios['raw'][0], 
											output_loc = output_loc,
											MIN_FREQ = MIN_FREQ, MAX_FREQ = MAX_FREQ,
											done = self.done_transform_audio, finished = self.finished_transform_audio)
		bk_thrd.start()
		self.add_metadata_item('cepstrum_rate', self.metadata['raw_rate'])


		

		

	def threshold_keystrokes(self, threshold):
		'''
			Thresholds the audio 
		'''
		#TODO 
		print('threshold_keystrokes')
		pass

	def cluster_keystrokes(self):
		'''
			K-means cluster the currently created keystrokes 
		'''
		#TODO 
		print('cluster_keystrokes')
		pass

	def predict_text(self):
		'''
			Run an hmm of the current clustered keystokes to predict text
		'''
		#TODO 
		print('predict_text')
		pass

	def get_random_keystoke(self, key = None):
		'''
			Returns the sound of a random keystroke, if key is None
			it will return a random sound, if key is a specific character,
			then a random keystroke of that character will be returned if it exists
		'''
		#TODO 
		pass

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


def createFileDialog(additionalExec = None, fileMode = QFileDialog.AnyFile, nameFilters = None, selectFilter = None, allowAll = True):
	'''
	  Bundles everything together needed to create a File chooser. This function 
	  returns a vbox and a QLineEdit input box. The vbox contains the QLineEdit input
	  box in addition to a QPushButton which when pressed, will launch a QFileDialog.
	  Once the user chooses their file, the result is stored in the QLinEdit input box.

	  Return values:
		 vbox : The box to be added to the form
		 inputBox : The reference needed to grab the file that was chosen

	  Arguments:
		 additionalExec : a function that will be run in addition to setting the QLineEdit
						  box upon completition of the QFileDialog
		 fileMode : Can be 1) QFileDialog.AnyFile - meaning that any file can be chosen or
						   2) QFileDialog.Directory - meaning that only directories can be chosen
		 nameFilters : When not None, this is a list of filters that can be added to the file dialog
		 selectFilter : If not None, the filter that should be selected by default
		 allowAll : When True, adds a filter for all file types by default
	'''

	# The input box that holds the chosen filename 
	inputBox = QLineEdit()

	def qLineFileBox(text):
		'''
		 Call additional exec if the input of the box is changed
		'''
		if additionalExec is not None:
			additionalExec(text)

	# Attatch change function to input box
	inputBox.textChanged.connect(qLineFileBox)

	# Dictionary that maps fileMode to button text
	modeToText = {
		QFileDialog.AnyFile : 'Choose File', 
		QFileDialog.Directory : 'Choose Folder'
	}

	# Ensure that only the correct fileModes are passed
	if fileMode not in modeToText:
		raise ValueError('The fileMode passed in to create a QFileDialog is not supported')
		return

	# Create push button to launch file chooser
	dialogPopup = QPushButton(modeToText[fileMode])

	# Function to call when button is pushed
	def popupFileChoose():
		'''
		 Callback of when dialogPopup button is pushed. This function will
		 launch a QFileDialog, place the result in the inputBox, and 
		 execute additionalExec(result) if necessary, passing into the function
		 the result of the file dialog
		'''
		# Create the fileDialog
		dlg = QFileDialog()

		# Set the mode to the parameter passed in
		dlg.setFileMode(fileMode)

		# Default filters
		filters = []
		if allowAll:
			filters.append("All (*)")

		# Set the filters if not None
		if nameFilters is not None:
			nameFilters.extend(filters)
			dlg.setNameFilters(nameFilters)

			# Set the default filter if not None
			if selectFilter is not None:
				dlg.selectNameFilter(selectFilter)
			elif len(filters) > 0:
				# Else set to default if there are any
				dlg.setNameFilters(filters)

		# Launch the file Dialog
		if dlg.exec_():
			# Get the results
			filenames = dlg.selectedFiles()
			sys.stdout = STDOUT

			# Only take the first result
			result = filenames[0]

			# Set the inputBox to the selected file
			inputBox.setText(result)

			# Call additional execution of necessary
			if additionalExec is not None:
				additionalExec(result)

	# Conenct the callback to the button
	dialogPopup.clicked.connect(popupFileChoose)

	# Create the layout
	vbox = QVBoxLayout()
	vbox.addWidget(inputBox)
	vbox.addWidget(dialogPopup)

	return vbox, inputBox

class TreeWidget(QTreeWidgetItem):
	def __init__(self, parent, values):
		super(self.__class__, self).__init__(parent, values)
		# parent branch
		self.parent = parent

		# values of branch
		self.values = values

class DecoderTreeWidget(QTreeWidgetItem):
	def __init__(self, parent, values, decoder):
		super(self.__class__, self).__init__(parent, values)
		# parent branch
		self.parent = parent

		# values of branch
		self.values = values

		# decoder for widget
		self.decoder = decoder

class AudioTreeWidget(QTreeWidgetItem):
	def __init__(self, parent, values, audio_files, video_files):
		super(self.__class__, self).__init__(parent, values)
		# parent branch
		self.parent = parent

		# values of branch
		self.values = values

		# decoder for widget
		self.audio_files = audio_files
		self.video_files = video_files


class ConvertVideoBackgroundThread(QThread):
	'''
		A generic background thread for our application 
	'''
	add_post = pyqtSignal(name='add_post')

	def __init__(self, parent, vid_types, input_file, audio_type,  add_post_callback):
		super(ConvertVideoBackgroundThread, self).__init__(parent)
		self.parent = parent
		self.vid_types = vid_types
		self.input_file = input_file
		self.audio_type = audio_type
		# Set up connection to send signal that updates progress bar
		self.add_post.connect(add_post_callback)

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

class VisualizeSignalBackgroundThread(QThread):
	'''
		A generic background thread for our application 
	'''
	add_post = pyqtSignal(name='add_post')

	def __init__(self, parent, signal_df, rate, outfile, audio_df, done):
		super(VisualizeSignalBackgroundThread, self).__init__(parent)
		self.parent = parent
		self.signal_df = signal_df
		self.rate = rate
		self.outfile = outfile
		self.audio_df = audio_df

		# Set up connection to send signal that updates progress bar
		self.add_post.connect(done)

	def run(self):
		'''
			Changes the status bar to say the parent tool is running 
			and then calls the process function
		'''

		#TODO 
		self.add_post.emit()
		spl.visualize_signal(signal_df = self.signal_df, rate = self.rate,
							outfile = self.outfile, audio_df = self.audio_df)

		self.add_post.emit()
		

		