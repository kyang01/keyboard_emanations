'''
   This is the main file that will launch the GUI
'''

# Get rid of annoying warnings on startup of GUI
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

import matplotlib
matplotlib.use('Qt5Agg')

import sys

# PyQT 
from PyQt5.QtWidgets import QApplication

# Launch our start window
from app.core.StartingWidgets import StartingWindow


def main():
    '''
      Main function that launches the GUI
    '''

    # Create the application
    app = QApplication(sys.argv)

    # Create a Starting Window
    start = StartingWindow(app)
    start.show()

    # Run Application
    sys.exit(app.exec_())

if __name__ == '__main__':    
    main()