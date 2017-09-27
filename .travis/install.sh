#!/bin/bash


set -e

if [[ $TRAVIS_OS_NAME == 'linux' ]]; then

    # Use apt-get
    sudo apt-get update;
    sudo apt-get upgrade;
fi

pip install -U coverage coveralls nose-exclude flake8 psutil scipy numpy scikit-learn;
pip install -r requirements.txt;
python setup.py install;
