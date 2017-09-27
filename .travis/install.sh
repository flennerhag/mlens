#!/bin/bash


set -e

if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
    sudo apt-get update;
fi

pip install -U coverage coveralls nose-exclude flake8 psutil scipy numpy scikit-learn;
pip install -r requirements.txt;
python setup.py install;
