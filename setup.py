from setuptools import setup, find_packages

long_description = """depynd is a Python library for evaluating dependencies among random variables from data.
It supports learning statistical dependencies for one-to-one, one-to-many, and many-to-many relationships,
where each one corresponds to

- mutual information (MI) estimation,
- feature selection, and
- graphical model structure learning,

respectively. Specifically, depynd supports MI estimation for discrete-continuous mixtures, MI-based feature
selection, and structure learning of undirected graphical models (a.k.a. Markov random fields)."""

setup(
    name='depynd',
    version='0.6.0',
    description='Evaluating dependencies among random variables.',
    long_description=long_description,
    author='Yuya Takashina',
    author_email='takashina2051@gmail.com',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    test_requires=['pytest', 'flake8'],
    url='https://github.com/y-takashina/depynd',
)
