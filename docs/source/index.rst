:py:mod:`depynd` - evaluating dependencies among random variables
=================================================================

:py:mod:`depynd` is a Python library for evaluating dependencies among random variables from data. It supports learning
statistical dependencies for one-to-one, one-to-many, and many-to-many relationships, where each one corresponds to

- :py:mod:`depynd.information`: mutual information (MI) estimation,
- :py:mod:`depynd.feature_selection`: feature selection, and
- :py:mod:`depynd.markov_networks`: Markov network structure learning,

respectively. Specifically, :py:mod:`depynd` supports MI estimation for discrete-continuous mixtures, MI-based feature
selection, and structure learning of Markov networks (a.k.a. Markov random fields).

