import numpy as np
from pytest import raises, fail

from depynd.information import mutual_information, conditional_mutual_information

X = np.random.multivariate_normal(np.zeros(2), np.eye(2), 10)
x = np.random.normal(0, 1, 10)
y = np.random.normal(0, 1, 20)
z = np.empty([10, 0])
estimators = ['dr', 'knn']


class TestMi:
    def test_knn(self):
        try:
            mutual_information(X, X, mi_estimator='knn', k=1)
            mutual_information(X, X, mi_estimator='knn', k=9)
        except (KeyError, ValueError):
            fail()
        with raises(AssertionError):
            mutual_information(X, X, mi_estimator='knn', k=0)
        with raises(AssertionError):
            mutual_information(X, X, mi_estimator='knn', k=10)
        with raises(AssertionError):
            mutual_information(X, X, mi_estimator='knn', k=0.1)

    def test_dr(self):
        try:
            mutual_information(X, X, mi_estimator='dr', sigma=1)
            mutual_information(X, X, mi_estimator='dr', n_bases=1)
            mutual_information(X, X, mi_estimator='dr', maxiter=1)
        except (KeyError, ValueError):
            fail()
        assert np.isnan(mutual_information(X, X, mi_estimator='dr', maxiter=1))
        with raises(AssertionError):
            mutual_information(X, X, mi_estimator='dr', sigma=0)
        with raises(AssertionError):
            mutual_information(X, X, mi_estimator='dr', n_bases=0)
        with raises(AssertionError):
            mutual_information(X, X, mi_estimator='dr', n_bases=0.1)
        with raises(AssertionError):
            mutual_information(X, X, mi_estimator='dr', maxiter=0)
        with raises(AssertionError):
            mutual_information(X, X, mi_estimator='dr', maxiter=0.1)

    def test_length(self):
        with raises(AssertionError):
            mutual_information(x, y)

    def test_dimension(self):
        try:
            mutual_information(x, x)
            mutual_information(x, X)
            mutual_information(X, x)
            mutual_information(X, X)
        except ValueError:
            fail()
        assert 0 == mutual_information(z, x)
        assert 0 == mutual_information(x, z)

    def test_mi_estimator(self):
        try:
            for estimator in estimators:
                mutual_information(x, x, mi_estimator=estimator)
        except ValueError:
            fail()
        with raises(ValueError):
            mutual_information(x, x, mi_estimator='')


class TestCmi:
    def test_length(self):
        with raises(AssertionError):
            conditional_mutual_information(x, x, y)
        with raises(AssertionError):
            conditional_mutual_information(x, y, x)
        with raises(AssertionError):
            conditional_mutual_information(y, x, x)

    def test_dimension(self):
        try:
            conditional_mutual_information(x, x, x)
            conditional_mutual_information(X, x, x)
            conditional_mutual_information(x, X, x)
            conditional_mutual_information(x, x, X)
            conditional_mutual_information(X, X, x)
            conditional_mutual_information(x, X, X)
            conditional_mutual_information(X, x, X)
            conditional_mutual_information(X, X, X)
        except ValueError:
            fail()
        assert 0 == conditional_mutual_information(z, x, x)
        assert 0 == conditional_mutual_information(x, z, x)
        assert mutual_information(x, x) == conditional_mutual_information(x, x, z)

    def test_mi_estimator(self):
        try:
            for estimator in estimators:
                conditional_mutual_information(x, x, x, mi_estimator=estimator)
        except ValueError:
            fail()
        with raises(ValueError):
            conditional_mutual_information(x, x, x, mi_estimator='')
