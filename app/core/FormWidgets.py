'''
    
'''
from .BackgroundThreads import *

class DisplayWidget(QWidget):
	'''
		General parent display widget
	'''
	def __init__(self, parent):
		QWidget.__init__(self)
		self.parent = parent
		self.buildUI()

	def buildUI(self):
		'''
			Builds the form
		'''
		fbox = QFormLayout()
		fbox.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

		fbox = self.build(fbox)

		self.setLayout(fbox)

	def build(self, fbox):
		'''
			override this method to edit the form fbox
		'''
		return fbox

class DecoderDisplay(QWidget):
	'''
		Form display for when a decoder is selected
	'''
	def __init__(self, parent):
		QWidget.__init__(self)
		self.parent = parent
		self.buildUI()

	def buildUI(self):
		'''
			Builds the display
		'''
		# We use the grid layout to create the ui
		grid = QGridLayout()
		grid.setSpacing(10)

		def display_metadata():
			'''
				displays the metadata key/value pairs
			'''
			grid.addWidget(QLabel("KEY:"), 0, 1)
			grid.addWidget(QLabel('VALUE'), 0, 2)
			x = 1
			for key in self.parent.metadata.keys():
				grid.addWidget(QLabel(key + ":"), x, 1)
				grid.addWidget(QLabel(self.parent.metadata[key]), x, 2)
				x += 1

		display_metadata()

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

		# predict the text of the keystrokes
		confuse_attacker = QPushButton("Confuse Attacker")
		confuse_attacker.clicked.connect(self.parent.confuse_attacker)
		grid.addWidget(confuse_attacker, 7, 0)
		

		self.setLayout(grid)

class MetadataDisplay(DisplayWidget):
	'''
		Form display for the add metadata widget
	'''
	def __init__(self, parent):
		DisplayWidget.__init__(self, parent)
		
	def build(self):
		'''
			Builds the form
		'''
		self.key_input = QLineEdit()
		fbox.addRow(QLabel("KEY:"), self.key_input)

		self.value_input = QLineEdit()
		fbox.addRow(QLabel("VALUE:"), self.value_input)

		add_button = QPushButton("Add to Metadata")
		add_button.clicked.connect(self.add_callback)
		fbox.addRow(add_button)

		return fbox

	def add_callback(self):
		key = str(self.key_input.text())
		value = str(self.value_input.text())

		self.parent.add_metadata_item(key, value)
		self.key_input.setText('')
		self.value_input.setText('')

	def closeEvent(self, event):
		self.parent.rebuild()

class VisSignalDisplay(DisplayWidget):
	'''
		Form display for the visualize signal button
	'''
	def __init__(self, parent, item, mapp):
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

		DisplayWidget.__init__(self, parent)
		
	def build(self):
		'''
			Builds the form
		'''

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

		# create the video
		add_button = QPushButton("Create Video")
		add_button.clicked.connect(self.submit_callback)
		fbox.addRow(add_button)

		return fbox

	def submit_callback(self):
		'''
			Callback for when the create video button is clicked
		'''
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
		self.mapp.status_label.setText('Transforming Audio Files...')

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
		self.mapp.status_label.setText('Thresholding...')

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
		self.mapp.status_label.setText('Transforming Audio Files...')



		bk_thrd = ClusterBackgroundThread(parent = self.parent.parent.parent, 
					save_dir = self.parent.threshold_directory, 
					rate = self.parent.metadata['raw_rate'], 
					done = self.parent.done_transform_audio, 
					finished = self.parent.finished_clustering, 
					MFCC_START = 0, MFCC_END = -1,
					winlen = 0.01, winstep = 0.0025, 
					numcep = 16, nfilt = 32, lowfreq = 400,
					highfreq = 12000, NUM_CLUSTERS = 40, N_COMPONENTS = 100)


		bk_thrd.start()
		self.close()

class PredictDisplay(QWidget):
	def __init__(self, parent, mapp):
		QWidget.__init__(self)
		self.parent = parent
		self.mapp = mapp

		# build ui
		self.buildUI()

	def buildUI(self):
		fbox = QFormLayout()
		fbox.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

		
		
		add_button = QPushButton("predict na")
		add_button.clicked.connect(self.submit_callback)
		fbox.addRow(add_button)

		self.setLayout(fbox)
	
	
	def submit_callback(self):
		
		self.mapp.progress_bar.setMaximum(3)
		self.mapp.status_label.setText('Transforming Audio Files...')


		self.close()

class ConfuseDisplay(QWidget):
	def __init__(self, parent, mapp):
		QWidget.__init__(self)
		self.parent = parent
		self.mapp = mapp

		# build ui
		self.buildUI()

	def buildUI(self):
		fbox = QFormLayout()
		fbox.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

		
		
		add_button = QPushButton("Confuse")
		add_button.clicked.connect(self.submit_callback)
		fbox.addRow(add_button)

		self.setLayout(fbox)
	
	
	def submit_callback(self):
		
		# self.mapp.progress_bar.setMaximum(3)
		self.mapp.status_label.setText('Confusing...')


		bk_thrd = DefenseBackgroundThread(self)
		bk_thrd.start()

