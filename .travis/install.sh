#!/bin/bash

set -e

if [[ $TRAVIS_OS_NAME == 'linux' ]]; then

    # Use apt-get
    sudo apt-get update;
fi

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then

    if [[ "$PY_VERSION" == "2.7" ]]; then

        brew update;
        brew install python27;
        virtualenv test-env -p python27;
        test-env/bin/activate

    fi

    if [[ "$PY_VERSION" == "3.5" ]]; then

        brew update;
        brew install python35;
        virtualenv test-env -p python35;
        test-env/bin/activate

    fi

    if [[ "$PY_VERSION" == "3.6" ]]; then

        brew update;
        brew install python36;
        virtualenv test-env -p python36;
        test-env/bin/activate

    fi
fi

pip install -U coverage coveralls nose-exclude flake8 psutil scipy numpy scikit-learn;
pip install -r requirements.txt;
python setup.py install;
