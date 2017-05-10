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
import pickle

# Configuration parameters
from app import config

import pandas as pd
import numpy as np
import ffmpy
import copy

sys.path.append('Libraries')
import signal_processing as spl 
import prediction_lib as pl

from .defense import DefenseBackgroundThread

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
		return pd.Series.from_csv(fname).to_dict()

	def add_metadata_item(self, key, value):
		metadata = pd.Series.from_csv(self.metadatafname)
		metadata.ix[key] = value
		metadata.to_csv(os.path.join(self.decode_folder, 'metadata.csv'))

	def add_metadata_item_callback(self):
		class MetadataDisplay(QWidget):
			def __init__(self, parent):
				QWidget.__init__(self)
				self.parent = parent
				self.buildUI()
				# self.close.connect(self.close_callback)

			def buildUI(self):
				fbox = QFormLayout()
				fbox.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

				self.key_input = QLineEdit()
				fbox.addRow(QLabel("KEY:"), self.key_input)

				self.value_input = QLineEdit()
				fbox.addRow(QLabel("VALUE:"), self.value_input)

				add_button = QPushButton("Add to Metadata")
				add_button.clicked.connect(self.add_callback)
				fbox.addRow(add_button)

				self.setLayout(fbox)

			def add_callback(self):
				key = str(self.key_input.text())
				value = str(self.value_input.text())

				self.parent.add_metadata_item(key, value)
				self.key_input.setText('')
				self.value_input.setText('')

			def closeEvent(self, event):
				self.parent.rebuild()


		self.w = MetadataDisplay(self)
		self.w.show()

	def rebuild(self):
		self.parent.parent.buildTree()

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

				# transform audio using fourier
				add_metadata_item = QPushButton("Add Metadata Item")
				add_metadata_item.clicked.connect(self.parent.add_metadata_item_callback)
				grid.addWidget(add_metadata_item, 0, 0)

				# transform audio using fourier
				visualize_signal = QPushButton("Visualize Signal")
				visualize_signal.clicked.connect(self.parent.visualize_signal)
				grid.addWidget(visualize_signal, 1, 0)

				# create a video of the currently selected audio
				create_video = QPushButton("Create Video")
				create_video.clicked.connect(self.parent.create_video)
				grid.addWidget(create_video, 2, 0)

				# transform audio using fourier
				transform_audio = QPushButton("Transform Audio")
				transform_audio.clicked.connect(self.parent.transform_audio)
				grid.addWidget(transform_audio, 3, 0)

				# thresold into keystrokes
				threshold_keystrokes = QPushButton("Threshold Keystrokes")
				threshold_keystrokes.clicked.connect(self.parent.threshold_keystrokes)
				grid.addWidget(threshold_keystrokes, 4, 0)

				# cluster keystrokes using kmeans
				cluster_keystrokes = QPushButton("Cluster Keystrokes")
				cluster_keystrokes.clicked.connect(self.parent.cluster_keystrokes)
				grid.addWidget(cluster_keystrokes, 5, 0)

				# predict the text of the keystrokes
				predict_text = QPushButton("Predict Text")
				predict_text.clicked.connect(self.parent.predict_text)
				grid.addWidget(predict_text, 6, 0)

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


		class VisSignalDisplay(QWidget):
			def __init__(self, parent, item, mapp):
				QWidget.__init__(self)
				self.parent = parent
				self.item = item
				self.mapp = mapp
				self.audio_type = self.item.values[0]

				# read in the signal df
				signal_floc = os.path.join(self.parent.signal_directory, '%s.csv' % self.audio_type)
				self.signal_df = pd.read_csv(signal_floc).set_index('time (s)')

				self.start_time, self.end_time = float(self.signal_df.index[0]), float(self.signal_df.index[-1])
				
				arr = np.arange(start = self.start_time, stop = np.ceil(self.end_time * 10) / 10, step =.1)
				self.times = list(map(str,arr))
				self.times.append(str(self.end_time))

				# build ui
				self.buildUI()

			def buildUI(self):
				fbox = QFormLayout()
				fbox.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

				# trim the start of the video
				self.start_trim = QComboBox(self)
				for time in self.times:
					self.start_trim.addItem(time)
				self.start_trim.setCurrentIndex(0)
				fbox.addRow(QLabel("Start Trim:"), self.start_trim)

				# trim the end of the video
				self.end_trim = QComboBox(self)
				for time in self.times:
					self.end_trim.addItem(time)
				self.end_trim.setCurrentIndex(len(self.times) - 1)
				fbox.addRow(QLabel("End Trim:"), self.end_trim)

				# get the max frames per minute
				self.max_f = QComboBox(self)
				for frm in range(3, 61):
					self.max_f.addItem(str(frm))
				self.max_f.setCurrentIndex(17)
				fbox.addRow(QLabel("Maximum Frames per minute:"), self.max_f)

				# get the frame break
				self.frame_b = QComboBox(self)
				for frm in range(1, 30):
					self.frame_b.addItem(str(frm))
				self.frame_b.setCurrentIndex(4)
				fbox.addRow(QLabel("Number of seconds per frame:"), self.frame_b)


				add_button = QPushButton("Create Video")
				add_button.clicked.connect(self.submit_callback)
				fbox.addRow(add_button)

				self.setLayout(fbox)

			def submit_callback(self):
				# location to save video
				output_loc = os.path.join(self.parent.video_directory, '%s_signal.mp4' % self.audio_type)
				
				# Update progress bar
				
				self.mapp.status_label.setText('Visualizing signal...')

				# determine the raw df
				if self.audio_type == 'raw':
					audio_df = self.signal_df
				else:
					audio_floc = os.path.join(self.parent.signal_directory, 'raw.csv')
					audio_df = pd.read_csv(audio_floc).set_index('time (s)')

				# get inputs
				MAX_FRAMES = int(self.max_f.currentText())
				_FRAME_BREAK = int(self.frame_b.currentText())
				START_TIME = float(self.start_trim.currentText())
				END_TIME = float(self.end_trim.currentText())



				self.signal_df = self.signal_df[(self.signal_df.index >= START_TIME) &  (self.signal_df.index <= END_TIME)]
				audio_df = audio_df[(audio_df.index >= START_TIME) &  (audio_df.index <= END_TIME)]

				num_sig = int(np.ceil((self.signal_df.index[-1] - self.signal_df.index[0]) / _FRAME_BREAK)) + 2
				self.mapp.progress_bar.setMaximum(num_sig)
				
				bk_thrd = VisualizeSignalBackgroundThread(parent = self.parent.parent.parent,
															signal_df = self.signal_df, 
															rate = int(self.parent.metadata['%s_rate' % self.audio_type]),
															outfile = output_loc,
															audio_df =  audio_df,
															MAX_FRAMES = MAX_FRAMES,
															_FRAME_BREAK = _FRAME_BREAK,
															START_TIME = START_TIME,
															END_TIME = END_TIME,
															done = self.parent.visualize_done,
															finished = self.parent.vis_fin)
				bk_thrd.start()

				self.close()


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
		


		class VideoDisplay(QWidget):
			def __init__(self, parent, item, mapp, vid_types):
				QWidget.__init__(self)
				self.parent = parent
				self.item = item
				self.mapp = mapp
				self.input_file = item.audio_files[0]
				self.audio_type = item.values[0]
				self.vid_types = vid_types


				# build ui
				self.buildUI()

			def buildUI(self):
				fbox = QFormLayout()
				fbox.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

				# To hold the selected files in the folder
				vbox = QVBoxLayout()
				self.vid_ts = QListWidget()
				self.vid_ts.setSelectionMode(QAbstractItemView.ExtendedSelection)
				vbox.addWidget(self.vid_ts)
				for key in self.vid_types:
					self.vid_ts.addItem(key)
				removeFile = QPushButton("Remove")
				removeFile.clicked.connect(partial(self.removeChosen, self.vid_ts))
				vbox.addWidget(removeFile)
				fbox.addRow(QLabel("Video Types to Create"), vbox)

				
				add_button = QPushButton("Create Video")
				add_button.clicked.connect(self.submit_callback)
				fbox.addRow(add_button)

				self.setLayout(fbox)
			
			def removeChosen(self, selectionBox):
				'''
					Remove selected items from selectionBox
				'''
				for selection in selectionBox.selectedItems():
					selectionBox.takeItem(selectionBox.row(selection))

			def submit_callback(self):
				self.mapp.progress_bar.setMaximum(len(self.vid_types))
				self.mapp.status_label.setText('Converting Audio files to video representation...')

				vid_types = {}
				for vid_type in [str(self.vid_ts.item(i).text()) for i in range(self.vid_ts.count())]:
					vid_types[vid_type] = self.vid_types[vid_type]
		
				bk_thrd = ConvertVideoBackgroundThread(self.parent.parent.parent, vid_types, self.input_file, self.audio_type, done = self.parent.done,
												finished = self.parent.fini)
				bk_thrd.start()

				self.close()

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
		class TransAudioDisplay(QWidget):
			def __init__(self, parent, mapp):
				QWidget.__init__(self)
				self.parent = parent
				self.mapp = mapp

				arr = np.arange(start = 20, stop = 20000, step =20)
				self.hz = list(map(str, arr))


		
				# build ui
				self.buildUI()

			def buildUI(self):
				fbox = QFormLayout()
				fbox.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

				# To hold the selected files in the folder
				self.transform_types = QComboBox()
				self.transform_types.addItem('fourier')
				fbox.addRow(QLabel("Type of transformation:"), self.transform_types)

				# To hold the selected files in the folder
				self.min_hz = QComboBox()
				for hz in self.hz:
					self.min_hz.addItem(hz)
				self.min_hz.setCurrentIndex(19)
				fbox.addRow(QLabel("Minimum Hz to keep:"), self.min_hz)

				# To hold the selected files in the folder
				self.max_hz = QComboBox()
				for hz in self.hz:
					self.max_hz.addItem(hz)
				self.max_hz.setCurrentIndex(len(self.hz) - 400)
				fbox.addRow(QLabel("Maximum Hz to keep:"), self.max_hz)


				
				add_button = QPushButton("Transform Audio")
				add_button.clicked.connect(self.submit_callback)
				fbox.addRow(add_button)

				self.setLayout(fbox)
			
			def removeChosen(self, selectionBox):
				'''
					Remove selected items from selectionBox
				'''
				for selection in selectionBox.selectedItems():
					selectionBox.takeItem(selectionBox.row(selection))

			def submit_callback(self):
				
				self.mapp.progress_bar.setMaximum(3)
				mapp.status_label.setText('Transforming Audio Files...')

				signal_name = str(self.transform_types.currentText())

				# The minimum frequency to keep for fourier
				MIN_FREQ = int(self.min_hz.currentText())
				

				# The maximum frequency to keep for fourier
				MAX_FREQ = int(self.max_hz.currentText())

				output_loc = os.path.join(self.parent.signal_directory, signal_name + '.csv')
				
				bk_thrd = TransformAudioBackgroundThread(parent = self.parent.parent.parent,
													audio_file = self.parent.audios['raw'][0], 
													output_loc = output_loc,
													MIN_FREQ = MIN_FREQ, MAX_FREQ = MAX_FREQ,
													done = self.parent.done_transform_audio, finished = self.parent.finished_transform_audio)
				bk_thrd.start()
				self.parent.add_metadata_item(signal_name + '_rate', self.parent.metadata['raw_rate'])



				self.close()
		mapp = self.parent.parent.parent.parent
		self.w = TransAudioDisplay(self, mapp)
		self.w.show()
	
	def threshold_keystrokes(self, threshold):
		'''
			Thresholds the audio 
		'''
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



		#TODO 
		class ThresholdDisplay(QWidget):
			def __init__(self, parent, mapp, audio_type):
				QWidget.__init__(self)
				self.parent = parent
				self.mapp = mapp
				self.audio_type = audio_type

				# read in the signal df
				signal_floc = os.path.join(self.parent.signal_directory, '%s.csv' % self.audio_type)
				self.signal_df = pd.read_csv(signal_floc).set_index('time (s)')


				self.start_time, self.end_time = float(self.signal_df.index[0]), float(self.signal_df.index[-1])
				
				arr = np.arange(start = self.start_time, stop = np.ceil(self.end_time * 10) / 10, step =.1)
				self.times = list(map(str,arr))
				self.times.append(str(self.end_time))

				# build ui
				self.buildUI()

			def buildUI(self):
				fbox = QFormLayout()
				fbox.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

				# trim the start of the video
				self.start_trim = QComboBox(self)
				for time in self.times:
					self.start_trim.addItem(time)
				self.start_trim.setCurrentIndex(0)
				fbox.addRow(QLabel("Start Trim:"), self.start_trim)

				# trim the end of the video
				self.end_trim = QComboBox(self)
				for time in self.times:
					self.end_trim.addItem(time)
				self.end_trim.setCurrentIndex(len(self.times) - 1)
				fbox.addRow(QLabel("End Trim:"), self.end_trim)

				mx_sig = self.signal_df['signal'].max() * 10
				print(mx_sig)
				# min thresh
				self.min_thresh_inp = QDoubleSpinBox(self)
				self.min_thresh_inp.setMinimum(0.01)
				self.min_thresh_inp.setValue(0.09)
				self.min_thresh_inp.setMaximum(mx_sig)
				self.min_thresh_inp.setSingleStep(0.01)
				fbox.addRow(QLabel('Minumum Threshold / 10 (on fourier) to be \nconsidered a keystroke:'), self.min_thresh_inp)

				# max thresh
				self.max_thresh_inp = QDoubleSpinBox(self)
				self.max_thresh_inp.setMinimum(0.01)
				self.max_thresh_inp.setMaximum(mx_sig)
				if mx_sig > 8.:
					self.max_thresh_inp.setValue(8.)
				else:
					self.max_thresh_inp.setValue(mx_sig)
				self.max_thresh_inp.setSingleStep(0.01)
				fbox.addRow(QLabel('Maximum Threshold / 10 (on fourier) to be \nconsidered a keystroke:'), self.max_thresh_inp)

				# mindist
				self.min_dist_inp = QSpinBox(self) 
				self.min_dist_inp.setMinimum(0)
				self.min_dist_inp.setMaximum(3000)
				self.min_dist_inp.setValue(30)
				fbox.addRow(QLabel('Minimum distance between keystrokes (ms):'), self.min_dist_inp)

				# key length
				self.key_len_inp = QSpinBox(self) 
				self.key_len_inp.setMinimum(0)
				self.key_len_inp.setMaximum(2000)
				self.key_len_inp.setValue(350)
				fbox.addRow(QLabel('The fixed length of a keystroke (ms):'), self.key_len_inp)

				# back prop
				self.back_prop_inp = QDoubleSpinBox(self) 
				self.back_prop_inp.setMinimum(0.01)
				self.back_prop_inp.setMaximum(.99)
				self.back_prop_inp.setValue(0.1)
				self.back_prop_inp.setSingleStep(0.01)
				fbox.addRow(QLabel('The proportion of a keystroke that occurs before the peak:'), self.back_prop_inp)


		
				
				add_button = QPushButton("Threshold")
				add_button.clicked.connect(self.submit_callback)
				fbox.addRow(add_button)

				self.setLayout(fbox)
			
			
			def submit_callback(self):
				
				self.mapp.progress_bar.setMaximum(11)
				mapp.status_label.setText('Thresholding...')

				MINT = float(self.start_trim.currentText())
				MAXT = float(self.end_trim.currentText())
				MIN_THRESH = float(self.min_thresh_inp.value()) / 10
				MAX_THRESH = float(self.max_thresh_inp.value()) / 10
				MIN_DIST = int(self.min_dist_inp.value())
				KEY_LEN = int(self.key_len_inp.value())
				BACK_PROP = float(self.back_prop_inp.value())
				TO_ADD = []
				save_dir = self.parent.threshold_directory

				fourier_floc = os.path.join(self.parent.signal_directory, 'fourier.csv')

				
				signal_floc =  os.path.join(self.parent.signal_directory, 'raw.csv')

				output_text = os.path.join(self.parent.decode_folder, 'output_text.txt')


				


				bk_thrd = ThresholdBackgroundThread(parent = self.parent.parent.parent, MINT = MINT,
										MAXT = MAXT, MIN_THRESH = MIN_THRESH, MAX_THRESH = MAX_THRESH,
										MIN_DIST = MIN_DIST, KEY_LEN = KEY_LEN, BACK_PROP = BACK_PROP,
										TO_ADD = TO_ADD,fourier_floc = fourier_floc, signal_floc = signal_floc,
											output_text = output_text,
											save_dir = save_dir, done = self.parent.done_transform_audio, finished = self.parent.finished_thresholding)
				bk_thrd.start()


				self.close()

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
		class ClusterBackgroundThread(QThread):
			'''
				A generic background thread for our application 
			'''
			add_post = pyqtSignal(name='add_post')

			def __init__(self, parent, save_dir, rate, done, finished, MFCC_START = 0, MFCC_END = -1,
						winlen = 0.01, winstep = 0.0025, numcep = 16, filt = 32, lowfreq = 400,
						highfreq = 12000, NUM_CLUSTERS = 40, N_COMPONENTS = 100):
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

		#TODO 
		class ClusterDisplay(QWidget):
			def __init__(self, parent, mapp):
				QWidget.__init__(self)
				self.parent = parent
				self.mapp = mapp

				# build ui
				self.buildUI()

			def buildUI(self):
				fbox = QFormLayout()
				fbox.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

				
				
				add_button = QPushButton("Cluster Keystrokes")
				add_button.clicked.connect(self.submit_callback)
				fbox.addRow(add_button)

				self.setLayout(fbox)
			
			
			def submit_callback(self):
				
				self.mapp.progress_bar.setMaximum(5)
				mapp.status_label.setText('Transforming Audio Files...')



				bk_thrd = ClusterBackgroundThread(parent = self.parent.parent.parent, 
							save_dir = self.parent.threshold_directory, 
							rate = self.parent.metadata['raw_rate'], 
							done = self.parent.done_transform_audio, 
							finished = self.parent.finished_clustering, 
							MFCC_START = 0, MFCC_END = -1,
							winlen = 0.01, winstep = 0.0025, 
							numcep = 16, filt = 32, lowfreq = 400,
							highfreq = 12000, NUM_CLUSTERS = 40, N_COMPONENTS = 100)


				bk_thrd.start()
				self.close()

		mapp = self.parent.parent.parent.parent
		self.w = ClusterDisplay(self, mapp)
		self.w.show()
		

	def predict_text(self):
		'''
			Run an hmm of the current clustered keystokes to predict text
		'''
		class ThresholdDisplay(QWidget):
			def __init__(self, parent, mapp):
				QWidget.__init__(self)
				self.parent = parent
				self.mapp = mapp

				# build ui
				self.buildUI()

			def buildUI(self):
				fbox = QFormLayout()
				fbox.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

				
				
				add_button = QPushButton("Create Video")
				add_button.clicked.connect(self.submit_callback)
				fbox.addRow(add_button)

				self.setLayout(fbox)
			
			
			def submit_callback(self):
				
				self.mapp.progress_bar.setMaximum(3)
				mapp.status_label.setText('Transforming Audio Files...')


				self.close()
		#TODO 
		# mapp = self.parent.parent.parent.parent
		# self.w = ThresholdDisplay(self, mapp)
		# self.w.show()
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
		

