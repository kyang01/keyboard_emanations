'''
    The StartingWindow class is a child of the QMainWindow class
    and is the main window of the application
'''

from .decoder import *
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
        
        # Create adjustable layout
        self.layout = QHBoxLayout(self)
        self.layout.addStretch(10)

        # Add the two widgets
        self.layout.addWidget(self.visualize)

        # Set the layout
        self.setLayout(self.layout)

class VisualizeWidget(QWidget):
    '''
        Widget apart of the MainWindow used to visualize images and
        hierarchy of experiment
    '''
    def __init__(self, parent = None, name = "Visualize Directory"):
        super(self.__class__, self).__init__(parent)
        # MainWindow
        self.Decoder = Decoder
        self.parent = parent 
        self.mapp = self.parent.parent
        self.processed = self.parent.parent.processed
        self.people_csv = os.path.join(self.processed, 'people.csv')
        self.tree = None
        self.process_stack = None

        if not os.path.exists(self.people_csv):
            df = pd.DataFrame(columns = config.PEOPLE_COLUMNS)
            df = df.append(pd.Series({'name' : "None", 'decoders' : ','}), ignore_index = True)
            df.to_csv(self.people_csv, index = False)

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
            self.decoder_list = self.buildTree()
            self.decoder_list.setMinimumWidth(400)
            # self.decoder_list.header().resizeSection(0, 150)
            itemsLayout.addWidget(self.decoder_list)

            # set layout
            itemsGroup.setLayout(itemsLayout)
            # itemsGroup.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
            grid.addWidget(itemsGroup, 0, 0, 2, 6)

        def manage_processing():
            '''
                Create the MDI space for random widgets
            '''

            self.process_stack = QStackedWidget()
            self.mdi = QMdiArea()
            self.process_stack.addWidget(self.mdi)

             # Add MDI area
            self.process_area = QGroupBox("", self)
            process_layout = QHBoxLayout(self.process_area)
            process_layout.addWidget(self.process_stack)
            self.process_area.setLayout(process_layout)
            # self.process_area.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
            grid.addWidget(self.process_area, 0, 6, 2, 6)

        def manage_display(external = True):
            '''
                Space to play audio/visual
            '''
            self.media_player = Player([], add_button = self.add_decoder)
            if external:
                self.media_player.setGeometry(0, 0, 800, 400)
                self.media_player.show()
            else:
                # Group sliders together
                videoGroup = QGroupBox("Display", self)
                videoLayout = QVBoxLayout(videoGroup)
                
                videoLayout.addWidget(self.media_player)
                videoGroup.setLayout(videoLayout)
                # videoGroup.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
                grid.addWidget(videoGroup, 3, 0, 2, 12)

        def add_horizontal_split():
            verticalLine    =  QFrame()
            verticalLine.setFrameStyle(QFrame.HLine)
            # verticalLine.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Expanding)

            grid.addWidget(verticalLine,2,0,1,12) 

        manage_decoders()
        manage_processing()
        # add_horizontal_split()
        manage_display()


        self.setLayout(grid) 
        
    def tree_change(self):
        '''
            When the item in the tree is changed
        '''
        # Grab the currently selected item
        item = self.tree.currentItem()

        # Display Decoder information
        if type(item) == DecoderTreeWidget :
            new_widget = item.decoder.update_display(self.process_area)
            self.process_stack.insertWidget(1, new_widget)
            self.process_stack.setCurrentIndex(1)

        elif type(item) == TreeWidget:
            self.process_stack.setCurrentIndex(0)
        elif type(item) == AudioTreeWidget:
            new_widget = item.decoder.update_display(self.process_area)
            self.process_stack.insertWidget(1, new_widget)
            self.process_stack.setCurrentIndex(1)

            self.media_player.playlist.clear()
            self.media_player.addToPlaylist(item.audio_files + item.video_files)
        elif self.process_stack:
            self.process_stack.setCurrentIndex(0)

    def buildTree(self):
        '''
            Builds the hierarchy tree to visualize the results 
        '''
        # Either create new tree or refresh clear the old tree
        if not self.tree:
            self.tree = QTreeWidget()

            # Callback for changing item in tree
            self.tree.itemSelectionChanged.connect(self.tree_change)
        else:
            self.tree.clear()
            self.tree.reset()
        self.tree_change()
        
        # Set the headers
        self.tree.setHeaderItem(QTreeWidgetItem(["People"]))
        
        # Create a branch for each plate and then recursively build
        self.people = {}
        people_dict = self.get_people_dict()
        for person in self.get_people():
            branch = TreeWidget(self.tree, [person])
            new_person = Person(self, person, people_dict[person])
            new_person.buildTree(branch)
            self.people[person] = new_person

        return self.tree

    def get_people(self):
        df = pd.read_csv(self.people_csv, index_col = False)
        return list(df['name'].values)

    def get_people_dict(self):
        df = pd.read_csv(self.people_csv, index_col = False)
        return df.set_index('name')['decoders'].to_dict()

    def add_person(self, person):
        df = pd.read_csv(self.people_csv, index_col = False)
        df = df.append(pd.Series({'name' : person, 'decoders' : ','}), ignore_index = True)
        df.to_csv(self.people_csv, index = False)

    def update_person(self, person, fold_num):
        df = pd.read_csv(self.people_csv, index_col = False)
        df = df.set_index('name')
        df.ix[person, 'decoders'] = ','.join(str(df.ix[person, 'decoders']).split(',')  + [str(fold_num)])
        df = df.reset_index()
        df.to_csv(self.people_csv, index = False)

    def add_decoder(self):
        '''
            Adds a new decoder object from the selected audio file
        '''

        # check to make sure an audio file is selected else throw error
        current_file = self.media_player.get_current_file()
        
        # Make sure a file was selected
        if not current_file: 
            QMessageBox.warning(self, 'No File Selected!',
                                            "Use 'Open Audio/Visual' and select an audio file in the list above" ,
                                            QMessageBox.Ok)
            return

        # Make sure we have a valid extension
        if os.path.splitext(current_file)[1].lower() not in config.AUDIO_EXTS:
            QMessageBox.warning(self, 'Invalid Extension!',
                                            "Valid audio file extensions:\n%s" % ' '.join(config.AUDIO_EXTS),
                                            QMessageBox.Ok)
            return

        


        self.w = CreateDecoderDisplay(self, self.mapp, self.tree.currentItem())
        self.w.show()

    def add_p(self):
        '''
            Updates the progress bar by 1
        '''
        self.mapp.progress_bar.setValue(self.mapp.progress_bar.value() + 1)

    def finished(self, display_text_main = 'Finished creating decoder!', display_text_small = 'Finished creating decoder!'):
        QMessageBox.information(self.mapp, display_text_main,
                                        display_text_small ,
                                        QMessageBox.Ok)
        self.buildTree()
        self.mapp.resetLabel()




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


        self.buildStatusBar()



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
    return [        ]
 