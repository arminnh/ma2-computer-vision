from distutils.core import setup
from Cython.Build import cythonize
import numpy

include_path = [numpy.get_include()]

setup(name='viterbi',
      ext_modules=cythonize("viterbi.pyx", include_path=include_path))