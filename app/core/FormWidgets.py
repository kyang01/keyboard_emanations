'''
    
'''
from .BackgroundThreads import *
from functools import partial

class DecoderDisplay(QWidget):
    '''
        Form display for when a decoder is selected
    '''
    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
        self.item = parent.parent.parent.tree.currentItem()
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
        if type(self.item) == AudioTreeWidget:
            visualize_signal = QPushButton("Visualize Signal")
            visualize_signal.clicked.connect(partial(self.parent.addWidg, button_to_formclass['visualize_signal']))
            grid.addWidget(visualize_signal, 1, 0)

        if type(self.item) == AudioTreeWidget and self.item.values[0] == 'raw':
            # create a video of the currently selected audio
            create_video = QPushButton("Create Video")
            create_video.clicked.connect(partial(self.parent.addWidg, button_to_formclass['create_video']))
            grid.addWidget(create_video, 2, 0)


        # transform audio using fourier
        transform_audio = QPushButton("Transform Audio")
        transform_audio.clicked.connect(partial(self.parent.addWidg, button_to_formclass['transform_audio']))
        grid.addWidget(transform_audio, 3, 0)

        if type(self.item) == AudioTreeWidget and self.item.values[0] == 'fourier':
            # thresold into keystrokes
            threshold_keystrokes = QPushButton("Threshold Keystrokes")
            threshold_keystrokes.clicked.connect(partial(self.parent.addWidg, button_to_formclass['threshold_keystrokes']))
            grid.addWidget(threshold_keystrokes, 4, 0)

            keystroke_inputloc = os.path.join(self.parent.threshold_directory, 'processing_input.csv')
            if os.path.exists(keystroke_inputloc):
                # cluster keystrokes using kmeans
                cluster_keystrokes = QPushButton("Cluster Keystrokes")
                cluster_keystrokes.clicked.connect(partial(self.parent.addWidg, button_to_formclass['cluster_keystrokes']))
                grid.addWidget(cluster_keystrokes, 5, 0)

                # predict the text of the keystrokes
                predict_text = QPushButton("Predict Text")
                predict_text.clicked.connect(partial(self.parent.addWidg, button_to_formclass['predict_text']))
                grid.addWidget(predict_text, 6, 0)

        # predict the text of the keystrokes
        confuse_attacker = QPushButton("Confuse Attacker")
        confuse_attacker.clicked.connect(partial(self.parent.addWidg, button_to_formclass['confuse_attacker']))
        grid.addWidget(confuse_attacker, 7, 0)
        
        self.setLayout(grid)

class DisplayWidget(QWidget):
    '''
        General parent display widget
    '''
    def __init__(self, parent, mapp, item, BackgroundThread = BackgroundThread, name = 'Running Task'):
        QWidget.__init__(self)
        self.parent = parent
        self.mapp = mapp 
        self.item = item
        self.name = name
        self.BackgroundThread = BackgroundThread
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

    def get_inputs(self):
        '''
            override this method to determine inputs for background thread
        '''
        return {}

    def submit_callback(self):
        # Update progress bar
        self.mapp.status_label.setText('%s...' % self.name)

        inputs = self.get_inputs()
        bk_thrd = self.BackgroundThread(parent = self.parent.parent.parent,
                                    done = self.parent.add_p, 
                                    finished = partial(self.parent.finished, 'Finished %s!' % self.name),
                                    inputs = inputs) 
        bk_thrd.start()

        self.close()

class CreateDecoderDisplay(DisplayWidget):
    def __init__(self, parent, mapp, item, BackgroundThread = CreateDecoderBackgroundThread, name = 'Creating Decoder'):
        DisplayWidget.__init__(self, parent, mapp, item, BackgroundThread, name)
        
    def build(self, fbox):
        self.current_file = self.parent.media_player.get_current_file()
        self.dir = os.path.dirname(self.current_file)

        # be able to match recording to a person
        self.people_stack = QStackedWidget()

        self.people_selected = QComboBox()
        self.people_selected.addItems(self.parent.get_people())

        #create new entry in people.csv
        self.people_stack.addWidget(self.people_selected)

        self.people_input = QLineEdit()
        self.people_stack.addWidget(self.people_input)
        fbox.addRow(QLabel('Person:'), self.people_stack)

        # Button to add new person
        self.new_person_button = QPushButton('New Person')
        self.new_person_button.clicked.connect(self.add_new)
        fbox.addRow(QLabel(''), self.new_person_button)

        # Choose input text
        sourceFileDialog, self.source_input_text = createFileDialog()
        input_text_fname = os.path.join(self.dir, 'input_text.txt')
        if os.path.exists(input_text_fname):
            self.source_input_text.setText(input_text_fname)
        fbox.addRow(QLabel("Input Text:"), sourceFileDialog)

        # Choose output text
        sourceFileDialog, self.source_output_text = createFileDialog()
        output_text_fname = os.path.join(self.dir, 'output_text.txt')
        if os.path.exists(output_text_fname):
            self.source_output_text.setText(output_text_fname)
        fbox.addRow(QLabel("Output Text:"), sourceFileDialog)

        # Choose output text
        sourceFileDialog, self.source_actual_text = createFileDialog()
        actual_text_fname = os.path.join(self.dir, 'actual_text.txt')
        if os.path.exists(actual_text_fname):
            self.source_actual_text.setText(actual_text_fname)
        fbox.addRow(QLabel("Actual Text:"), sourceFileDialog)

        # Show if the file has metadata it can extract
        self.metadata = self.parent.Decoder.extract_metadata(self.current_file)
        metadata_str = textwrap.fill(json.dumps(self.metadata), 50)
        fbox.addRow(QLabel("Metadata:"), QLabel(metadata_str))

        # submit button
        create_decoder = QPushButton('Create Decoder')
        create_decoder.clicked.connect(self.submit_callback)
        fbox.addRow(create_decoder)

        return fbox

    def add_new(self):
        '''
        '''
        self.people_stack.setCurrentIndex(1)

    def get_inputs(self):
        self.mapp.progress_bar.setMaximum(8)

        def get_person():
            # determine if we are creating a new person
            person_ind = self.people_stack.currentIndex()

            # old person
            if person_ind == 0:
                person = str(self.people_selected.currentText())

            # new person
            elif person_ind == 1:
                person = str(self.people_input.text())

                if person in self.parent.get_people():
                    QMessageBox.warning(self, 'Person exists!',
                                        "A person with this name already exists, please change the person's name" ,
                                        QMessageBox.Ok)
                    return

                self.parent.add_person(person)

            # error
            else:
                assert(False)

            return person
        
        def validate(fil):
            if os.path.exists(fil) and os.path.splitext(fil)[1] == '.txt':
                return fil

        inputs = {
            'current_file' : self.current_file,
            'input_text' : validate(str(self.source_input_text.text())),
            'output_text' : validate(str(self.source_output_text.text())),
            'actual_text' : validate(str(self.source_actual_text.text())),
            'person' : get_person(),
            'mapp' : self.mapp,
            'parent' : self.parent,
            'decoder' : self.parent.Decoder
        }

        return inputs

class MetadataDisplay(DisplayWidget):
    '''
        Form display for the add metadata widget
    '''
    def __init__(self, parent, mapp, item):
        DisplayWidget.__init__(self, parent, mapp, item)
        
    def build(self, fbox):
        '''
            Builds the form
        '''
        # Determine the key
        self.key_input = QLineEdit()
        fbox.addRow(QLabel("KEY:"), self.key_input)

        # Determine the value
        self.value_input = QLineEdit()
        fbox.addRow(QLabel("VALUE:"), self.value_input)

        # button for callback
        add_button = QPushButton("Add to Metadata")
        add_button.clicked.connect(self.add_callback)
        fbox.addRow(add_button)

        return fbox

    def add_callback(self):
        '''
            Callback to adding metadata item
        '''
        # add keypair
        key, value = str(self.key_input.text()), str(self.value_input.text())
        self.parent.add_metadata_item(key, value)

        # reset
        self.key_input.setText('')
        self.value_input.setText('')

    def closeEvent(self, event):
        '''
            Make parent build on close
        '''
        self.parent.rebuild()

class VisSignalDisplay(DisplayWidget):
    '''
        Form display for the visualize signal button
    '''
    def __init__(self, parent, mapp, item, BackgroundThread = VisualizeSignalBackgroundThread, name = 'Creating Videos to Visualize Signals'):
        self.min_MAX_FRAMES, self.max_MAX_FRAMES, self.val_MAX_FRAMES = 3, 61, 17
        self.min_FRAME_BREAK, self.max_FRAME_BREAK, self.val_FRAME_BREAK = 1, 30, 4
        DisplayWidget.__init__(self, parent, mapp, item, BackgroundThread, name)

    def build(self, fbox):
        '''
            Builds the form
        '''
        # trim the start of the video
        self.start_trim = QDoubleSpinBox(self)
        self.start_trim.setMinimum(float(self.parent.metadata['raw_start_time']))
        self.start_trim.setMaximum(float(self.parent.metadata['raw_end_time']))
        self.start_trim.setValue(float(self.parent.metadata['raw_start_time']))
        fbox.addRow(QLabel("Start Trim:"), self.start_trim)

        # trim the end of the video
        self.end_trim = QDoubleSpinBox(self)
        self.end_trim.setMinimum(float(self.parent.metadata['raw_start_time']))
        self.end_trim.setMaximum(float(self.parent.metadata['raw_end_time']))
        self.end_trim.setValue(float(self.parent.metadata['raw_end_time']))
        fbox.addRow(QLabel("End Trim:"), self.end_trim)

        # get the max frames per minute
        self.max_f = QComboBox(self)
        for frm in range(self.min_MAX_FRAMES, self.max_MAX_FRAMES):
            self.max_f.addItem(str(frm))
        self.max_f.setCurrentIndex(self.val_MAX_FRAMES)
        fbox.addRow(QLabel("Maximum Frames per minute:"), self.max_f)

        # get the frame break
        self.frame_b = QComboBox(self)
        for frm in range(self.min_FRAME_BREAK, self.max_FRAME_BREAK,):
            self.frame_b.addItem(str(frm))
        self.frame_b.setCurrentIndex(self.val_FRAME_BREAK,)
        fbox.addRow(QLabel("Number of seconds per frame:"), self.frame_b)

        # create the video
        add_button = QPushButton("Create Video")
        add_button.clicked.connect(self.submit_callback)
        fbox.addRow(add_button)

        return fbox

    def get_inputs(self):
        '''
            Callback for when the create video button is clicked
        '''
        audio_type = self.item.values[0]
        
        # read in signal_df
        signal_floc = os.path.join(self.parent.signal_directory, '%s.csv' % audio_type)
        signal_df = pd.read_csv(signal_floc).set_index('time (s)')

        # determine the audio df
        if audio_type == 'raw':
            audio_df = signal_df.copy()
        else:
            audio_floc = os.path.join(self.parent.signal_directory, 'raw.csv')
            audio_df = pd.read_csv(audio_floc).set_index('time (s)')

        # determine start and end times
        START_TIME, END_TIME = float(self.start_trim.value()), float(self.end_trim.value())
        
        # shrink to subset of df
        signal_df = signal_df[(signal_df.index >= START_TIME) &  (signal_df.index <= END_TIME)]
        audio_df = audio_df[(audio_df.index >= START_TIME) &  (audio_df.index <= END_TIME)]

        # determine the number of signals sent for a video this long
        _FRAME_BREAK = int(self.frame_b.currentText())
        num_sig = int(np.ceil((signal_df.index[-1] - signal_df.index[0]) / _FRAME_BREAK)) + 2
        self.mapp.progress_bar.setMaximum(num_sig)
        
        inputs = {
            'signal_df' : signal_df,
            'rate' : int(self.parent.metadata['%s_rate' % audio_type]),
            'outfile' : os.path.join(self.parent.video_directory, '%s_signal.mp4' % audio_type),
            'audio_df' :  audio_df,
            'audio_type' : audio_type,
            'MAX_FRAMES' : int(self.max_f.currentText()),
            '_FRAME_BREAK' : _FRAME_BREAK,
            'START_TIME' : START_TIME,
            'END_TIME' : END_TIME,

        }
        return inputs

class VideoDisplay(DisplayWidget):
    def __init__(self, parent, mapp, item, BackgroundThread = ConvertVideoBackgroundThread, name = 'Creating Audio Files to Video'):
        DisplayWidget.__init__(self, parent, mapp, item, BackgroundThread, name)
        
    def build(self, fbox):
        '''
            builds the form
        '''
        self.vid_types = self.parent.vid_types

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

        return fbox
    
    def removeChosen(self, selectionBox):
        '''
            Remove selected items from selectionBox
        '''
        for selection in selectionBox.selectedItems():
            selectionBox.takeItem(selectionBox.row(selection))

    def get_inputs(self):

        vid_types = {}
        for vid_type in [str(self.vid_ts.item(i).text()) for i in range(self.vid_ts.count())]:
            vid_types[vid_type] = self.vid_types[vid_type]

        self.mapp.progress_bar.setMaximum(len(vid_types))

        inputs = {
            'vid_types' : vid_types, 
            'input_file' : self.item.audio_files[0], 
            'audio_type' : self.item.values[0],
        }
        return inputs

class TransAudioDisplay(DisplayWidget):
    def __init__(self, parent, mapp, item, BackgroundThread = TransformAudioBackgroundThread, name = 'Transforming Audio'):
        self.start_hz, self.end_hz, self.skip_hz, self.val_min_hz, self.val_max_hz = 20, 20000, 20, 400, 12000
        DisplayWidget.__init__(self, parent, mapp, item, BackgroundThread, name)
        
    def build(self, fbox):
        # To hold the selected files in the folder
        self.transform_types = QComboBox()
        self.transform_types.addItem('fourier')
        fbox.addRow(QLabel("Type of transformation:"), self.transform_types)

        arr = np.arange(start = self.start_hz, stop = self.end_hz, step = self.skip_hz)
        self.hz = list(map(str, arr))

        # To hold the selected files in the folder
        self.min_hz = QDoubleSpinBox()
        self.min_hz.setMinimum(self.start_hz)
        self.min_hz.setMaximum(self.end_hz)
        self.min_hz.setSingleStep(self.skip_hz)
        self.min_hz.setValue(self.val_min_hz)
        fbox.addRow(QLabel("Minimum Hz to keep:"), self.min_hz)

        # To hold the selected files in the folder
        self.max_hz = QDoubleSpinBox()
        self.max_hz.setMinimum(self.start_hz)
        self.max_hz.setMaximum(self.end_hz)
        self.max_hz.setSingleStep(self.skip_hz)
        self.max_hz.setValue(self.val_max_hz)
        fbox.addRow(QLabel("Maximum Hz to keep:"), self.max_hz)
        
        add_button = QPushButton("Transform Audio")
        add_button.clicked.connect(self.submit_callback)
        fbox.addRow(add_button)

        return fbox
    
    def removeChosen(self, selectionBox):
        '''
            Remove selected items from selectionBox
        '''
        for selection in selectionBox.selectedItems():
            selectionBox.takeItem(selectionBox.row(selection))

    def get_inputs(self):
        
        self.mapp.progress_bar.setMaximum(3)
        signal_name = str(self.transform_types.currentText())
        self.parent.add_metadata_item(signal_name + '_rate', self.parent.metadata['raw_rate'])
        
        inputs = {
            'audio_file' : self.parent.audios['raw'][0], 
            'output_loc' : os.path.join(self.parent.signal_directory, signal_name + '.csv'),
            'MIN_FREQ' : int(self.min_hz.value()), 
            'MAX_FREQ' : int(self.max_hz.value()),
            'parent' : self.parent
        }
        return inputs

class ThresholdDisplay(DisplayWidget):
    def __init__(self, parent, mapp, item, BackgroundThread = ThresholdBackgroundThread, name = 'Thresholding Keystrokes'):
        self.mn_sig  = 0.01
        self.step = 0.01
        self.mn_val = 0.09
        self.md_min, self.md_max, self.md_val = 0, 3000, 30
        self.kl_min, self.kl_max, self.kl_val = 0, 2000, 350
        self.bp_min, self.bp_max, self.bp_val = 0.01, 0.99, 0.1

        DisplayWidget.__init__(self, parent, mapp, item, BackgroundThread, name)

    def build(self, fbox):

        self.mx_sig = float(self.parent.metadata['max-fourier-signal']) * 10
        if self.mx_sig > 8.:
            self.mx_val = 8.
        else:
            self.mx_val = self.mx_sig
        
        

        # trim the start of the video
        self.start_trim = QDoubleSpinBox(self)
        self.start_trim.setMinimum(float(self.parent.metadata['raw_start_time']))
        self.start_trim.setMaximum(float(self.parent.metadata['raw_end_time']))
        self.start_trim.setValue(float(self.parent.metadata['raw_start_time']))
        fbox.addRow(QLabel("Start Trim:"), self.start_trim)

        # trim the end of the video
        self.end_trim = QDoubleSpinBox(self)
        self.end_trim.setMinimum(float(self.parent.metadata['raw_start_time']))
        self.end_trim.setMaximum(float(self.parent.metadata['raw_end_time']))
        self.end_trim.setValue(float(self.parent.metadata['raw_end_time']))
        fbox.addRow(QLabel("End Trim:"), self.end_trim)

        
        # min thresh
        self.min_thresh_inp = QDoubleSpinBox(self)
        self.min_thresh_inp.setMinimum(self.mn_sig)
        self.min_thresh_inp.setMaximum(self.mx_sig)
        self.min_thresh_inp.setValue(self.mn_val)
        self.min_thresh_inp.setSingleStep(self.step)
        fbox.addRow(QLabel('Minumum Threshold / 10 (on fourier) to be \nconsidered a keystroke:'), self.min_thresh_inp)

        # max thresh
        self.max_thresh_inp = QDoubleSpinBox(self)
        self.max_thresh_inp.setMinimum(self.mn_sig)
        self.max_thresh_inp.setMaximum(self.mx_sig)
        self.max_thresh_inp.setValue(self.mn_val)
        self.max_thresh_inp.setSingleStep(self.step)
        fbox.addRow(QLabel('Maximum Threshold / 10 (on fourier) to be \nconsidered a keystroke:'), self.max_thresh_inp)

        # mindist
        self.min_dist_inp = QSpinBox(self) 
        self.min_dist_inp.setMinimum(self.md_min)
        self.min_dist_inp.setMaximum(self.md_max)
        self.min_dist_inp.setValue(self.md_val)
        fbox.addRow(QLabel('Minimum distance between keystrokes (ms):'), self.min_dist_inp)

        # key length
        self.key_len_inp = QSpinBox(self) 
        self.key_len_inp.setMinimum(self.kl_min)
        self.key_len_inp.setMaximum(self.kl_max)
        self.key_len_inp.setValue(self.kl_val)
        fbox.addRow(QLabel('The fixed length of a keystroke (ms):'), self.key_len_inp)

        # back prop
        self.back_prop_inp = QDoubleSpinBox(self) 
        self.back_prop_inp.setMinimum(self.bp_min)
        self.back_prop_inp.setMaximum(self.bp_max)
        self.back_prop_inp.setValue(self.bp_val)
        self.back_prop_inp.setSingleStep(self.step)
        fbox.addRow(QLabel('The proportion of a keystroke that occurs before the peak:'), self.back_prop_inp)

        add_button = QPushButton("Threshold")
        add_button.clicked.connect(self.submit_callback)
        fbox.addRow(add_button)

        return fbox
    
    
    def submit_callback(self):
        self.mapp.progress_bar.setMaximum(12)
        
        inputs = {
            'MINT' : float(self.start_trim.value()),
            'MAXT' : float(self.end_trim.value()), 
            'MIN_THRESH' : float(self.min_thresh_inp.value()) / 10, 
            'MAX_THRESH' : float(self.max_thresh_inp.value()) / 10,
            'MIN_DIST' : int(self.min_dist_inp.value()), 
            'KEY_LEN' : int(self.key_len_inp.value()), 
            'BACK_PROP' : float(self.back_prop_inp.value()),
            'TO_ADD' : [],
            'fourier_floc' : os.path.join(self.parent.signal_directory, 'fourier.csv'), 
            'signal_floc' : os.path.join(self.parent.signal_directory, 'raw.csv'),
            'output_text' : os.path.join(self.parent.decode_folder, 'output_text.txt'),
            'save_dir' : self.parent.threshold_directory, 
        }

        bk_thrd = ThresholdBackgroundThread(parent = self.parent.parent.parent,
                                    done = self.parent.add_p, 
                                    finished = self.parent.finished_thresholding,
                                    inputs = inputs) 
        bk_thrd.start()
        self.close()

class ClusterDisplay(DisplayWidget):
    def __init__(self, parent, mapp, item, BackgroundThread = ClusterBackgroundThread, name = 'Clustering Keystrokes'):
        DisplayWidget.__init__(self, parent, mapp, item, BackgroundThread, name)
    
    def build(self, fbox):
        '''
            builds the form
        '''
        
        add_button = QPushButton("Cluster Keystrokes")
        add_button.clicked.connect(self.submit_callback)
        fbox.addRow(add_button)

        return fbox
    
    
    def get_inputs(self):
        
        self.mapp.progress_bar.setMaximum(5)

        inputs = {
            'save_dir' : self.parent.threshold_directory, 
            'rate' : self.parent.metadata['raw_rate'],
            'MFCC_START' : 0, 
            'MFCC_END' : -1,
            'winlen' : 0.01, 
            'winstep' : 0.0025, 
            'numcep' : 16, 
            'nfilt' : 32, 
            'lowfreq' : 400,
            'highfreq' : 12000,
            'NUM_CLUSTERS' : 40,
            'N_COMPONENTS' : 100
        }

        return inputs

class PredictDisplay(DisplayWidget):
    def __init__(self, parent, mapp, item, BackgroundThread = PredictBackgroundThread, name = 'Predicting Text'):
        DisplayWidget.__init__(self, parent, mapp, item, BackgroundThread, name)
    

    def build(self, fbox):
        '''
            builds the form
        '''
        
        add_button = QPushButton("predict na")
        add_button.clicked.connect(self.submit_callback)
        fbox.addRow(add_button)

        return fbox
    
    
    def get_inputs(self):
        
        self.mapp.progress_bar.setMaximum(3)
        inputs = {}
        return inputs

class ConfuseDisplay(DisplayWidget):
    def __init__(self, parent, mapp, item, BackgroundThread = DefenseBackgroundThread, name = 'Confusing'):
        DisplayWidget.__init__(self, parent, mapp, item, BackgroundThread, name)

    def build(self, fbox):
        add_button = QPushButton("Confuse")
        add_button.clicked.connect(self.submit_callback)
        fbox.addRow(add_button)

        return fbox
    
    def get_inputs(self):
        
        self.mapp.progress_bar.setMaximum(3)

        inputs = {}
        return inputs

button_to_formclass = {
    'visualize_signal' : VisSignalDisplay,
    'transform_audio' : TransAudioDisplay,
    'threshold_keystrokes' : ThresholdDisplay,
    'create_video' : VideoDisplay,
    'cluster_keystrokes' : ClusterDisplay,
    'predict_text' : PredictDisplay,
    'confuse_attacker' : ConfuseDisplay
}