"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT

ML-Ensemble - a library for Ensemble Learning
"""

from setuptools import setup, find_packages
import mlens

VERSION = mlens.__version__

setup(name='mlens',
      version=VERSION,
      description='Machine Learning Ensemble Library',
      author='Sebastian Flennerhag',
      author_email='sebastianflennerhag@hotmail.com',
      url='https://github.com/flennerhag/mlens',
      packages=find_packages(),
      package_data={'': ['LICENSE.txt',
                         'README.md',
                         'requirements.txt'
                         ]
                    },
      include_package_data=True,
      install_requires=['numpy>=1.11',
                        'scipy>=0.17'],
      license='MIT',
      platforms='any',
      classifiers=['License :: OSI Approved :: MIT License',
                   'Development Status :: 4 - Beta',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'
                   ],
      long_description="""
A library for memory efficient parallelized Ensemble learning

Documentation available at:
    - http://mlens.readthedocs.io/en/latest/
    - https://github.com/flennerhag/mlens

Contact
=======
For questions and comments reach out to sebastianflennerhag@hotmail.com.

This project is hosted at https://github.com/flennerhag/mlens
""")
