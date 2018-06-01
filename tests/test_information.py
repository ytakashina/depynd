import numpy as np
from pytest import raises, fail

from depynd.information import mi_dr, mi_knn, mutual_information, conditional_mutual_information

mean = np.zeros(2)
cov = [[1, 0.5], [0.5, 1]]
X = np.random.multivariate_normal(mean, cov, 1000)
y, z = X.T


class TestMiKnn:

    def test_k(self):
        try:
            mi_knn(X, X, k=1)
        except:
            fail()
        with raises(ValueError):
            mi_knn(X, X, k=0.1)
        with raises(ValueError):
            mi_knn(X, X, k=-1)
        with raises(ValueError):
            mi_knn(X, X, k=[1])


class TestMi:
    def test_1d_1d(self):
        try:
            mutual_information(y, z)
        except:
            fail()

    def test_1d_2d(self):
        try:
            mutual_information(y, X)
            mutual_information(X, y)
        except:
            fail()

    def test_2d_2d(self):
        try:
            mutual_information(X, X)
        except:
            fail()
