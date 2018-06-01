import numpy as np
from pytest import raises, fail

from depynd.information import mi_dr, mi_knn, mutual_information, conditional_mutual_information

mean = np.zeros(2)
cov = [[1, 0.5], [0.5, 1]]
X = np.random.multivariate_normal(mean, cov, 1000)
y, z = X.T


class TestMiKnn:

    def test_1d_1d(self):
        mi_knn(y, z, k=3)

    def test_1d_2d(self):
        mi_knn(y, X, k=3)
        mi_knn(X, y, k=3)

    def test_2d_2d(self):
        mi_knn(X, X, k=3)

    def test_k(self):
        try:
            mi_knn(y, z, k=1)
        except:
            fail()
        with raises(ValueError):
            mi_knn(y, z, k=0.1)
        with raises(ValueError):
            mi_knn(y, z, k=-1)
        with raises(ValueError):
            mi_knn(y, z, k=[1])
