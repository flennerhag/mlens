#!/bin/bash

set -e

if [[ $TRAVIS_OS_NAME == 'linux' ]]; then

    # Use apt-get
    sudo apt-get update;
fi

pip install coverage coveralls nose-exclude flake8 psutil scikit-learn;
pip install -r requirements.txt;
python setup.py install;

echo "Installation complete"