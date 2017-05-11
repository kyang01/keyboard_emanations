# keyboard_emanations

Exploring unsupervised learning techniques for predicting keystrokes presses only using the raw audio. We fourier transform the audio, and simple thresholding to create keystrokes. We then use kmeans++ to cluster the keytrokes and use the clusters as the emissions in a hidden markov model, combined with the viterbi algorithm, to predict text. 

Included is a GUI to process audio.

 [ ADD YOUTUBE CLIP HERE]

## Structure
```
├── app                     # Contains code to create the GUI
├── audio                     # Contains example audio files
├── data	                  # Folder to hold various datafiles
├── img                     # Contains various images
├── Libraries               # Folder to hold libraries to process signals 
├── models                   # Folder to hold hmm models that performed well
├── notebooks               # IPython notebooks with visualizations of process and example use
├── papers                   # Relevant literature
├── process                   # Folder holding results from gui 
├── results                     # Results of parameter tuning
├── text                     # Actual text files here
├── venv                     # Virtual environment
├── videos                     # Folder to hold various videos
├── app.py                   # starts the gui application
├── print_ind.py            # Code to label keystrokes in realtime
├── start.sh            # Bash script that installs dependencies for GUI
├── README.md
└──.gitignore	
```

## GUI

The GUI can be launched using the bash script start.sh

On a mac, open up terminal.
```
    cd keyboard_emanations
    bash start.sh
```

The start.sh script should install any missing dependencies, and then launch the application.

## Dependencies 

The bash script start.sh should automatically install all dependencies on a mac. 
```
    bash start.sh
```

The dependencies are:

[Python 3.5.*> and pip](https://www.continuum.io/downloads)

virtualenv:
```
    pip3 install virtualenv
```

ffmpeg

```
    # install brew
    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

    # brew install ffmpeg
    brew install ffmpeg --with-fdk-aac --with-ffplay --with-freetype --with-frei0r --with-libass --with-libvo-aacenc --with-libvorbis --with-libvpx --with-opencore-amr --with-openjpeg --with-opus --with-rtmpdump --with-schroedinger --with-speex --with-theora --with-tools
```

## Data

Note: Much of the data we used is missing due to file contraints of github. 

## Notebooks

PreProcessing.ipynb : The notebook tjat was used for peak detection and data exploration.

- Categorizes key press data
- Creates output for the Processing notebook
- Builds video showing key presses with overlayed text

Processing.ipynb : The notebook that was used to tune the text prediction algorithms

- Clusters detected keystrokes using kmeans 
- Uses bigrams combined with kmeans clusters to construct an hmm to predict mappings of keystroaks to characters 
- Has functionality for testing a variety of parameters easily, stored to the results folder

Compare Noise.ipynb

- Has functionality to test what presence of white noise will disrupt the algorithm


Notebooks can be viewed using ipython and jupyer notebook

```
	cd keyboard_emanations
	jupyter notebook
```


