# GUI tools
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *



# File systems
import os, sys, glob, time, shutil
STDOUT = sys.stdout
from functools import partial
import json
import itertools

# Configuration parameters
from app import config

import pandas as pd


class Decoder(object):
	'''
		Decoder object that will decode and predict text from
		an audio recording. If labels are attached, it will use
		them for validation purposes
	'''

	def __init__(self, audio_file, input_text = None, 
					output_text = None, actual_text = None, person = None):
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

		# Holds the various audio representations in memory
		self.audios = {}

		# The location of video files for various of audio files
		self.videos = {}

		# Extract whatever metadata possible from the filenames
		self.metadata = self.extract_metadata(self.audio_file)

	@classmethod
	def extract_metadata(self, fname):
		'''
			extracts metadata from file names for later analyeses
		'''
		#TODO
		print(fname)
		return {}

	def save(self):
		'''
			Save the decoder object to the processing folder
		'''
		#TODO
		pass

	def assign_person(self):
		'''
			Assigns decoder to a specific person
		'''
		self.person = person

	def create_video(self, video_type):
		'''
			Creates a video of an audio recording
		'''
		#TODO 
		pass

	def transform_audio(self, audio_type):
		'''
			Transforms the raw audio into another representation
		'''
		#TODO 
		pass

	def threshold_audio(self, threshold):
		'''
			Thresholds the audio 
		'''
		#TODO 
		pass

	def create_keystokes(self):
		'''
			Creates the keystrokes from the currently loaded audio
		'''
		#TODO 
		pass

	def cluster_keystrokes(self):
		'''
			K-means cluster the currently created keystrokes 
		'''
		#TODO 
		pass

	def predict_text(self):
		'''
			Run an hmm of the current clustered keystokes to predict text
		'''
		#TODO 
		pass

	def visualize(self):
		'''
			Visualize keystrokes
		'''
		#TODO 
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

	def __init__(self, name, decoder = None):
		self.name = name
		self.decoders = []
		if decoder:
			self.add_decoder(decoder)

	def add_decoder(self, decoder):
		'''
			Adds a decoder to the current person
		'''
		# add to the list of decoders
		self.decoders.append(decoder)

		# save to the process folder for later access

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

class Keystroke(object):
	'''
		A keystroke 
	'''
	def __init__(self, raw_rep, transformed_rep):
		self.representation = {
			'raw' : raw_rep,
			'transformed' : transformed_rep
		}
		self.label = None
		self.distribution = None

	def add_label(self, label):
		'''
			Adds a label to the keystroke
		'''
		self.label = label

	def add_distribution(self, dist):
		'''
			Adds a distribution over what the keystoke 
			may be
		'''
		self.distribution = dist





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
