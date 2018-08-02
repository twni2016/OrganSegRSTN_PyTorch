from distutils.core import setup, Extension
import numpy as np
import os
import shutil


os.environ['CC'] = 'g++';

setup(name = 'fast cpp functions', version = '1.0', \
    ext_modules = [Extension('_fast_functions', \
    ['fast_functions.cc', 'fast_functions.i'], include_dirs = [np.get_include(), '.'])])

#shutil.copyfile('build/lib.linux-x86_64-2.7/_fast_functions.so', '_fast_functions.so')
