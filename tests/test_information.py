import numpy as np
from pytest import raises, fail

from depynd.information import mi_dr, mi_knn, mutual_information, conditional_mutual_information

mean = np.zeros(2)
cov = [[1, 0.5], [0.5, 1]]
X = np.random.multivariate_normal(mean, cov, 10)
x = np.random.normal(0, 1, 10)
y = np.random.normal(0, 1, 20)


class TestMi:
    def test_k(self):
        try:
            mutual_information(X, X, k=1)
            mutual_information(X, X, k=9)
        except:
            fail()
        with raises(AssertionError):
            mutual_information(X, X, k=10)
        with raises(AssertionError):
            mutual_information(X, X, k=0.1)
        with raises(AssertionError):
            mutual_information(X, X, k=-1)
        with raises(AssertionError):
            mutual_information(X, X, k=[1])

    def test_length(self):
        with raises(AssertionError):
            mutual_information(x, y)

    def test_dimension(self):
        try:
            mutual_information(x, x)
            mutual_information(x, X)
            mutual_information(X, x)
            mutual_information(X, X)
        except:
            fail()
