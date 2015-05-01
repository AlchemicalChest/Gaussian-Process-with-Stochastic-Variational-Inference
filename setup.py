#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup

# Version number
version = '0.0.1'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name = 'GPSVI',
      version = version,
      author = 'Ziang Zhu',
      author_email = 'zhu.ziang.1990@gmail.com',
      description = ('Gaussian Process with Stochastic Variational Inference'),
      license = 'MIT',
      keywords = 'machine-learning gaussian-processes stochastic variational inference',
      url = 'https://github.com/AlchemicalChest/Gaussian-Process-with-Stochastic-Variational-Inference',
      packages = ['GPSVI.core',
                  'GPSVI.util',
                  'GPSVI.test'],
      package_dir={'GPSVI': 'GPSVI'},
      include_package_data = True,
      py_modules = ['GPSVI.__init__'],
      long_description=read('README.md'),
      install_requires=['numpy>=1.7', 'scipy>=0.12'],
      extras_require = {'docs':['matplotlib >=1.3','Sphinx','IPython']}
      )