import numpy as np
from pytest import raises, fail

from depynd.feature_selection import select

X = np.random.multivariate_normal(np.zeros(2), np.eye(2), 10)
x = np.random.normal(0, 1, 10)
y = np.random.normal(0, 1, 20)
z = np.empty([10, 0])


class TestSelect:
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
