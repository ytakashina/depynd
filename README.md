# depynd [![Build Status](https://travis-ci.com/y-takashina/depynd.svg?branch=master)](https://travis-ci.com/y-takashina/depynd) [![Documentation Status](https://readthedocs.org/projects/depynd/badge/?version=latest)](https://depynd.readthedocs.io/en/latest/?badge=latest)

`depynd` is a Python library for evaluating dependencies among random variables from data. It supports learning
statistical dependencies for one-to-one, one-to-many, and many-to-many relationships, where each one corresponds to

- mutual information (MI) estimation,
- feature selection, and
- graphical model structure learning,

respectively. Specifically, `depynd` supports MI estimation for discrete-continuous mixtures, MI-based feature selection, and
structure learning of undirected graphical models (a.k.a. Markov random fields).

Here is the [documentation](https://depynd.readthedocs.io/).

## Dependencies
- Python (>=3.5)
- NumPy (>=1.13.0)
- SciPy
- scikit-learn

## Installation
```
$ pip install depynd
```

## How to use
See [notebooks](https://github.com/y-takashina/depynd/tree/master/notebooks).
