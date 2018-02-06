from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='bayesian_networks',
    ext_modules=cythonize("bayesian_networks.pyx"),
)
