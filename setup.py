#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
date: 10/01/2017
license: MIT
mlens - an ML Ensemble learning library
"""

from setuptools import setup, find_packages
import mlens

VERSION = mlens.__version__

setup(name='mlens',
      version=VERSION,
      description='Machine Learning Ensemble Library',
      author='Sebastian Flennerhag',
      author_email='sebastianflennerhag@gmail.com',
      url='https://github.com/flennerhag/mlens',
      packages=find_packages(),
      package_data={'': ['LICENSE.txt',
                         'README.md',
                         'requirements.txt']},
      include_package_data=True,
      install_requires=['numpy>=1.11.0', 'scipy>=0.17', 'scikit-learn>=0.18.1'],
      license='MIT',
      platforms='any',
      classifiers=['License :: OSI Approved :: MIT License',
                   'Development Status :: 3 - Alpha',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      long_description="""
A library for automated, paralellized Ensemble learning

Contact
=============
If you have any questions or comments about mlens,
do not hesitate to reach out!
email: sebastianflennerhag@gmail.com
This project is hosted at https://github.com/flennerhag/mlens
""")
