# Ensure Python 3 exists
if command -v python3 > /dev/null 2>&1; then
    PYTHON=python3
    echo Python 3 installed
else
    if [ -d ~/anaconda3 ]; then
      source ~/anaconda3/bin/activate
      PYTHON=python
    else
      echo Missing Python 3
      echo Downloading from https://www.continuum.io/downloads...

      if command -v wget > /dev/null 2>&1; then
          echo wget installed
      else
        # Ensure brew exists
        if command -v brew > /dev/null 2>&1; then
            echo brew exists
        else
            echo installing brew...
            /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
        fi
        echo installing wget...
        brew install wget
      fi
      wget https://repo.continuum.io/archive/Anaconda3-4.3.1-MacOSX-x86_64.sh
      bash Anaconda3-4.3.1-MacOSX-x86_64.sh -b 
      source ~/anaconda3/bin/activate
      if [ -d ~/anaconda3 ]; then
        source ~/anaconda3/bin/activate
        PYTHON=python
      else
        source ~/anaconda/bin/activate
        PYTHON=python
      fi
    fi
fi

# Ensure virtualenv exists
if command -v virtualenv > /dev/null 2>&1; then
    echo virtualenv exists
else
  if command -v pip3 > /dev/null 2>&1; then
    pip3 install virtualenv
  else
    pip install virtualenv
  fi
fi

# check if virtual environment has been created before
if [ ! -d "venv" ]; then
  virtualenv -p $PYTHON venv
  # activate the environment
  source venv/bin/activate
  pip install -r requirements.txt
else
  # activate the environment
  source venv/bin/activate
fi

# Ensure ffmpeg exists
if command -v ffmpeg > /dev/null 2>&1; then
    echo ffmpeg exists
else
    # Ensure brew exists
    if command -v brew > /dev/null 2>&1; then
        echo brew exists
    else
        echo installing brew...
        /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    fi
    brew install ffmpeg --with-fdk-aac --with-ffplay --with-freetype --with-frei0r --with-libass --with-libvo-aacenc --with-libvorbis --with-libvpx --with-opencore-amr --with-openjpeg --with-opus --with-rtmpdump --with-schroedinger --with-speex --with-theora --with-tools
fi

# run the gui
$PYTHON app.py
