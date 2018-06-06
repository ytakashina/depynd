import numpy as np
from pytest import raises, fail

from depynd.feature_selection import select

X = np.random.multivariate_normal(np.zeros(2), np.eye(2), 10)
x = np.random.normal(0, 1, 10)
y = np.random.normal(0, 1, 20)
z = np.empty([10, 0])
methods = ['mifs', 'mrmr']


class TestSelect:
    def test_lamb(self):
        try:
            select(X, x, lamb=1.0)
            select(X, x, lamb=1)
            select(X, x, lamb=0)
            select(X, x, lamb=-1.0)
            select(X, x, lamb=np.inf)
            select(X, x, lamb=-np.inf)
        except ValueError:
            fail()
        with raises(ValueError):
            select(X, x, lamb=None, k=None)
        with raises(AssertionError):
            select(X, x, lamb=[0])

    def test_k(self):
        with raises(AssertionError):
            select(X, x, lamb=None, k=3)
        with raises(AssertionError):
            select(X, x, lamb=None, k=0)

    def test_dimension(self):
        try:
            select(X, x)
            select(X, x[:, None])
        except ValueError:
            fail()
        with raises(ValueError):
            select(x, x)
        with raises(ValueError):
            select(x[:, None], x)
        with raises(ValueError):
            select(X, X)
        with raises(ValueError):
            select(X, z)

    def test_length(self):
        with raises(ValueError):
            select(X, y)
        with raises(ValueError):
            select(X[:1], x[:1])

    def test_method(self):
        try:
            for method in methods:
                select(X, x, method=method)
        except ValueError:
            fail()
        with raises(ValueError):
            select(X, x, method='')
