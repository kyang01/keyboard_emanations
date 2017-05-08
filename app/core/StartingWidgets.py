'''
    The StartingWindow class is a child of the QMainWindow class
    and is the main window of the application
'''

from .misc import *
from .VideoWidget import *

class MainWindow(QWidget):

    def __init__(self, open_loc = None):
        super(self.__class__, self).__init__()

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
        
        # Setup the UI for the visualize widget
        self.setupUi()

    def setupUi(self):
        '''
            Builds the form that contains the options necessary 
            to determine how to convert the images
        '''
        grid = QGridLayout()
        grid.setSpacing(10)

        # Add widget to hold an image
         # Object view

        self.itemsGroup = QGroupBox("Samples", self)
        self.itemsLayout = QVBoxLayout(self.itemsGroup)
        self.itemsLayout.addWidget(QListWidget())
        self.itemsLayout.addWidget(QPushButton("Add Current Audio"))
        self.itemsGroup.setLayout(self.itemsLayout)
        grid.addWidget(self.itemsGroup, 0, 0, 2, 2)

        self.itemsLayout = QVBoxLayout(self.itemsGroup)
        self.itemsLayout.addWidget(QListWidget())
        self.itemsGroup.setLayout(self.itemsLayout)
        grid.addWidget(self.itemsGroup, 1, 0, 1, 2)

        # Add MDI area
        self.mdi = QMdiArea(self)
        self.mdiArea = QGroupBox("", self)
        self.mdiLayout = QHBoxLayout(self.mdiArea)
        self.mdiLayout.addWidget(self.mdi)
        self.mdiArea.setLayout(self.mdiLayout)
        grid.addWidget(self.mdiArea, 0, 2, 2, 4)

        # Group sliders together
        self.videoGroup = QGroupBox("Display", self)
        self.videoLayout = QVBoxLayout(self.videoGroup)
        self.videoLayout.addWidget(Player([]))
        self.videoGroup.setLayout(self.videoLayout)
        grid.addWidget(self.videoGroup, 2, 0, 10, 6)
        
        self.setLayout(grid) 

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
 