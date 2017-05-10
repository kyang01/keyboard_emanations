'''
    
'''
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


