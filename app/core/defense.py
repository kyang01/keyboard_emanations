# Defends from this attack by playing random keyboard emanations as you type.
# Goal is to cause enough noise in the data to throw off predictions.

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from pynput import keyboard
import simpleaudio as sa
from numpy import random
import time



class DefenseBackgroundThread(QThread):
    '''
        Plays keyboard sounds to interfere with the detection algorithm.
        Start and End are the only important public API functions
    '''
    # end = pyqtSignal(name='end')
    def __init__(self, parent):
        super(DefenseBackgroundThread, self).__init__(parent)
        self.parent = parent
        self.defending = False
        self.interfering = False
        
        # Set up sound files
        key_sound = sa.WaveObject.from_wave_file("app/core/assets/KeyPress.wav")
        space_sound = sa.WaveObject.from_wave_file("app/core/assets/SpacePress.wav")
        multi_sound = sa.WaveObject.from_wave_file("app/core/assets/FastKeys.wav")
        self.sounds = [key_sound, space_sound, multi_sound]

    def run(self):
        pass
    
    def startDefense(self):
        print("start")
        self.defending = True
        # Listen to keyboard
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def endDefense(self):
        print("stop")
        self.defending = False
        keyboard.Listener.StopException

    # Only detects special keys (shift, option...) due to OS X security
    def on_press(self, key):
        if not self.interfering:
            self.playInterference()

    def playInterference(self):
        self.interfering = True
        play_at = random.exponential(0.1)
        
        # print("play at " + str(play_at))
        
        # Wait, then play sound
        time.sleep(play_at)
        
        # Play sound
        wav_obj = random.choice(self.sounds, p=[0.4,0.4,0.2])
        play_obj = wav_obj.play()
        play_obj.wait_done()
        
        # 60% possibility of recurrance
        if random.rand() > 0.4:
            # Continue keystrokes recursively
            self.playInterference()
        else:
            self.interfering = False
            