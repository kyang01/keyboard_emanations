# keyboard_emanations

Exploring unsupervised learning techniques for predicting keystrokes presses only using the raw audio. 

## Structure
```
├── data	                  # Folder to hold various datafiles
├── libraries               # Folder to hold self created libraries 
├── models                   # Folder to hold hmm models that performed well
├── papers                   # Relevant literature
├── results                     # Results of parameter tuning
├── text                     # Actual text files here
├── Kevins Notebook.ipynb	  
├── LABLED_CHAR_INPUTS_COMBINED_RESULTS-51.7%.csv
├── PreProcessing.ipynb  
├── Processing.ipynb
├── README.md
└──.gitignore	
```

Note: Much of the data we used is missing due to file contraints of github. 

PreProcessing.ipynb : The notebook tjat was used for peak detection and data exploration.

- Categorizes key press data
- Creates output for the Processing notebook
- Builds video showing key presses with overlayed text

Processing.ipynb : The notebook that was used to tune the text prediction algorithms

- Clusters detected keystrokes using kmeans 
- Uses bigrams combined with kmeans clusters to construct an hmm to predict mappings of keystroaks to characters 
- Has functionality for testing a variety of parameters easily, stored to the results folder
- Has functionality to test what presence of white noise will disrupt the algorithm

## Dependencies 

matplotlib.animation among others (ffmpeg) 

TODO - Coming Soon
