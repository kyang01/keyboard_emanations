# Ensure Python 3 exists
if command -v python3 > /dev/null 2>&1; then
    echo Python 3 installed
else
    echo Missing Python 3
    echo Downloading from https://www.continuum.io/downloads
    wget https://repo.continuum.io/archive/Anaconda3-4.3.1-MacOSX-x86_64.sh
    bash Anaconda3-4.3.1-MacOSX-x86_64.sh -y
fi

# Ensure virtualenv exists
if command -v virtualenv > /dev/null 2>&1; then
    echo virtualenv exists
else
    pip3 install virtualenv
fi

# check if virtual environment has been created before
if [ ! -d "venv" ]; then
  virtualenv -p python3 venv
  # activate the environment
  source venv/bin/activate
  pip install -r requirements.txt
else
	# activate the environment
	source venv/bin/activate
fi

# run the gui
python app.py
