'''
    The StartingWindow class is a child of the QMainWindow class
    and is the main window of the application
'''

from .misc import *
from .VideoWidget import *

class MainWindow(QWidget):

    def __init__(self, parent = None):
        super(self.__class__, self).__init__(parent)
        self.parent = parent

        # Build GUI
        self.setupUi()

    def setupUi(self):
        '''
         Sets up the UI for the main Window
        '''
        # Widget dealing with visualizing 
        self.visualize = VisualizeWidget(self)
        self.mdi = self.visualize.mdi
        
        # Create adjustable layout
        self.layout = QHBoxLayout(self)
        self.layout.addStretch(10)

        # Add the two widgets
        self.layout.addWidget(self.visualize)

        # Set the layout
        self.setLayout(self.layout)

    def addWidg(self, WidgClass, **kwargs):
        '''
            Adds a widget to the the mdi area
        '''
        # Create new widget
        sub_widget = WidgClass(self, **kwargs)

        # Add the window and display it
        self.mdi.addSubWindow(sub_widget)

        # Show the new widget
        sub_widget.show()

class VisualizeWidget(QWidget):
    '''
        Widget apart of the MainWindow used to visualize images and
        hierarchy of experiment
    '''
    def __init__(self, parent = None, name = "Visualize Directory"):
        super(self.__class__, self).__init__(parent)
        # MainWindow
        self.parent = parent 
        self.processed = self.parent.parent.processed
        self.people_csv = os.path.join(self.processed, 'people.csv')

        if not os.path.exists(self.people_csv):
            pd.DataFrame(columns = config.PEOPLE_COLUMNS).to_csv(self.people_csv, index_col = False)

        self.decoders = []
        self.people = []
        
        # Setup the UI for the visualize widget
        self.setupUi()

    def setupUi(self):
        '''
            Builds the form that contains the options necessary 
            to determine how to convert the images
        '''
        # We use the grid layout to create the ui
        grid = QGridLayout()
        grid.setSpacing(10)

        def manage_decoders():
            '''
                Have a QListWidget to manage the audio decoders 
            '''
            # create layout
            itemsGroup = QGroupBox("Decoders", self)
            itemsLayout = QVBoxLayout(itemsGroup)

            # List of decoders
            self.decoder_list = QListWidget()
            itemsLayout.addWidget(self.decoder_list)

            # button to add decoder
            add_decoder_btn = QPushButton("Add New Decoder from Selected Audio")
            add_decoder_btn.clicked.connect(self.add_decoder)            
            itemsLayout.addWidget(add_decoder_btn)

            # set layout
            itemsGroup.setLayout(itemsLayout)
            grid.addWidget(itemsGroup, 0, 0, 2, 2)

        def manage_mdi():
            '''
                Create the MDI space for random widgets
            '''
             # Add MDI area
            self.mdi = QMdiArea(self)
            mdiArea = QGroupBox("", self)
            mdiLayout = QHBoxLayout(mdiArea)
            mdiLayout.addWidget(self.mdi)
            mdiArea.setLayout(mdiLayout)
            grid.addWidget(mdiArea, 0, 2, 2, 4)

        def manage_display():
            '''
                Space to play audio/visual
            '''
            # Group sliders together
            videoGroup = QGroupBox("Display", self)
            videoLayout = QVBoxLayout(videoGroup)
            self.media_player = Player([])
            videoLayout.addWidget(self.media_player)
            videoGroup.setLayout(videoLayout)
            grid.addWidget(videoGroup, 2, 0, 10, 6)

        manage_decoders()
        manage_mdi()
        manage_display()

        self.setLayout(grid) 

    def add_decoder(self):
        '''
            Adds a new decoder object from the selected audio file
        '''

        # check to make sure an audio file is selected else throw error
        current_file = self.media_player.get_current_file()
        
        # Make sure a file was selected
        if not current_file: 
            QMessageBox.warning(self, 'No File Selected!',
                                            "Select an audio file from the bottom list" ,
                                            QMessageBox.Ok)
            return

        # Make sure we have a valid extension
        if os.path.splitext(current_file)[1].lower() not in config.AUDIO_EXTS:
            QMessageBox.warning(self, 'Invalid Extension!',
                                            "Valid audio file extensions:\n%s" % ' '.join(config.AUDIO_EXTS),
                                            QMessageBox.Ok)
            return
         

        class CreateDecoder(QWidget):
            def __init__(self, parent, current_file):
                QWidget.__init__(self)
                self.parent = parent
                self.current_file = current_file
                print(current_file)

                self.buildUI()


            def buildUI(self):
                # create the form
                fbox = QFormLayout()
                fbox.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

                # be able to match recording to a person
                self.people_stack = QStackedWidget()

                self.people_selected = QComboBox()
                self.people_selected.addItem('None')
                self.people_stack.addWidget(self.people_selected)

                self.people_input = QLineEdit()
                self.people_stack.addWidget(self.people_input)
                fbox.addRow(QLabel('Person:'), self.people_stack)

                # Button to add new person
                self.new_person_button = QPushButton('New Person')
                self.new_person_button.clicked.connect(self.add_new)
                fbox.addRow(QLabel(''), self.new_person_button)

                # Choose input text
                sourceFileDialog, self.source_input_text = createFileDialog(additionalExec = self.update_input_text)
                fbox.addRow(QLabel("Input Text:"), sourceFileDialog)

                # Choose output text
                sourceFileDialog, self.source_ouput_text = createFileDialog(additionalExec = self.update_output_text)
                fbox.addRow(QLabel("Output Text:"), sourceFileDialog)

                # Choose output text
                sourceFileDialog, self.source_actual_text = createFileDialog(additionalExec = self.update_actual_text)
                fbox.addRow(QLabel("Actual Text:"), sourceFileDialog)

                # Show if the file has metadata it can extract
                metadata = Decoder.extract_metadata('test')
                metadata_str = json.dumps(metadata)
                fbox.addRow(QLabel("Metadata:"), QLabel(metadata_str))

                # submit button
                create_decoder = QPushButton('Create Decoder')
                create_decoder.clicked.connect(self.add_decoder)
                fbox.addRow(create_decoder)

                # set the layout
                self.setLayout(fbox)

            def add_decoder(self):

                # determine if we are creating a new person
                person_ind = self.people_stack.currentIndex()

                # old person
                if person_ind == 0:
                    person = str(self.people_selected.currentText())

                # new person
                elif person_ind == 1:
                    person = str(self.people_input.text())

                    #create new entry in people.csv


                # error
                else:
                    assert(False)

                print(person)

                pass

            def add_new(self):
                self.people_stack.setCurrentIndex(1)


            def update_input_text(self, val):

                print(val)
                
            def update_output_text(self, val):

                print(val)
                
            def update_actual_text(self, val):

                print(val)
                


        self.w = CreateDecoder(self, current_file)
        self.w.show()



        # create popup that asks for additional files to attach, metadata to pull

        # create decoder object

        # save to process folder

        # load the decoders to the list view
        self.load_decoders()

        

    def load_decoders(self):
        '''
            Loads any decoders from the process folder
        '''
        # check for decoders in process folder

        # load into list view
        pass

    def load_people(self):
        '''
            Loads person objects into memory
        '''
        # find person objects in processing folder

        # loads into memory and displays
        pass

    def add_person(self):
        '''
            Adds a new person 
        '''
        pass

class StartingWindow(QMainWindow):
    '''
        Class for the main window. Handles the creation of the status bar,
        and menu bar. Initally, the help widget is used as the central widget 
        and then once an experiment is loaded, the mdi region of MainWindow 
        is used as the centralWidget
    '''
    
    def __init__(self, app):
        super(self.__class__, self).__init__()
        
        # Main application
        self.app = app
        
        # Working directory
        self.home = os.getcwd()

        # Directory to save processed files
        self.processed = os.path.join(self.home, 'process')

        if not os.path.exists(self.processed):
            os.mkdir(self.processed)

        # Setup the interface
        self.setupUI()

    ################################################################
    # Setup user interface: status bar, menu, shortcuts
    ################################################################


    def setupUI(self):
        '''
            Sets up the user interface for the starting window 
        '''
        # No status bar to start
        self.status_bar = None

        # Title widget
        self.setWindowTitle(config.APP_NAME)  

        # Set central widget to help menu
        self.setCentralWidget(MainWindow(self))

        # Setup the various bars at the top of the screen
        self.buildMenu()

        # maximize screen
        # self.showMaximized()

    def setupAction(self, action_tuple, menu):
        '''
            Add an action to the menu with the options specified
            by the options dictionary
        '''
        # None indicates a separator
        if not action_tuple:
            menu.addSeparator()
            return

        # Parse apart tuple
        action, options = action_tuple

        # Create action from options
        action_item = QAction(action, self)        
        action_item.triggered.connect(options['callback'])
        # action_item.setShortcut(options['shortcut'])
        
        menu.addAction(action_item)
        options['object'] = action_item

    def buildMenu(self):
        '''
         Builds the top menu bar
        '''
        # Create a dictionary of menu items from the config file
        self.menu_items = GET_MENU_BAR(self)

        # Create a menu bar
        self.bar = self.menuBar()
        self.bar.setNativeMenuBar(False)

        for menu_item, actions in self.menu_items:
            # Create new menu dropdown
            new_menu = self.bar.addMenu(menu_item)
            
            # Add the elements according to config dict, None indicates separator
            [self.setupAction(action_tuple, new_menu) for action_tuple in actions]
        
    ################################################################
    # View Callback
    ################################################################

    def cascade(self):
        '''  
         Cascade the windows in the top left corner
        '''
        self.centralWidget().mdi.cascadeSubWindows()

    def tile(self):
        '''
         Tile the windows to fit the main window
        '''
        self.centralWidget().mdi.tileSubWindows()

    ################################################################
    # Tools Callback
    ################################################################
    
    def addWidg(self, WidgClass, **kwargs):
        '''
            Adds a widget of type WidgClass to the MainWindow mdi 
        '''
        # Calls the function within the MainWindow
        self.centralWidget().addWidg(WidgClass, **kwargs)

    ################################################################
    # Status bar
    ################################################################

    def resetLabel(self):
        '''
            Resets the label in the status bar
        '''
        self.status_label.setText(config.APP_NAME)
        self.progress_bar.setMaximum(0)
        self.progress_bar.setMaximum(1)

    def buildStatusBar(self):
        '''
            Builds a status bar at the bottom of the main window which will include
            a label that indicates the current job being run, a progress bar that
            shows the status of that job, and a stop button to kill the current job
        '''
        if self.status_bar:
            self.status_bar.show()
        else:
            # Add a status bar
            self.status_bar = QStatusBar(self)
            self.setStatusBar(self.status_bar)

            # Create a label for the status bar
            self.status_label = QLabel(self)
            self.status_bar.addWidget(self.status_label)

            # Add a progress bar to the statusBar
            self.progress_bar = QProgressBar(self)
            self.progress_bar.setProperty("value", 0)
            self.progress_bar.setMaximum(1)
            self.status_bar.addWidget(self.progress_bar)

            self.resetLabel()

            # Add a stop button for the current process
            self.btn_stop = QPushButton(self)
            self.btn_stop.setEnabled(False)
            self.btn_stop.setText("Stop")
            self.status_bar.addWidget(self.btn_stop)

def GET_MENU_BAR(self):
    '''
        Given an instance of a MainWindow, function returns a datastructure
        correpsonding to the menu bar at the top of the applications. We have a list of
        tuples (MENU_NAME, MENU_ITEMS)

        where each ITEM in MENU_ITEMS contains a 
            -shortcut
            -callback function (within self)
    '''
    return [
            ('View', [
                ('Cascade', {
                    'shortcut' : 'Ctrl+Alt+C',
                    'callback' : self.cascade,
                }),
                ('Tiled', {
                    'shortcut' : 'Ctrl+Alt+T',
                    'callback' : self.tile,
                }),
            ]),
        ]
 