import numpy as np
from pytest import raises, fail, approx

from depynd.information import mutual_information, conditional_mutual_information

X = np.random.multivariate_normal(np.zeros(2), np.eye(2), 10)
w = np.random.randint(0, 2, 10)
x = np.random.normal(0, 1, 10)
y = np.random.normal(0, 1, 20)
z = np.empty([10, 0])


class TestMi:
    def test_auto(self):
        with raises(AssertionError):
            mutual_information(x, x, mi_estimator='dr', is_discrete=True)
        with raises(AssertionError):
            mutual_information(x, x, mi_estimator='plugin', is_discrete=False)
        with raises(AssertionError):
            mutual_information(x, x, mi_estimator='plugin', is_discrete='auto')
        with raises(AssertionError):
            mutual_information(w, w, mi_estimator='dr', is_discrete='auto')
        try:
            mutual_information(x, x, mi_estimator='auto', is_discrete='auto')
            mutual_information(x, w, mi_estimator='auto', is_discrete='auto')
            mutual_information(w, w, mi_estimator='auto', is_discrete='auto')
            mutual_information(x, x, mi_estimator='auto', is_discrete=True)
            mutual_information(x, x, mi_estimator='auto', is_discrete=False)
            mutual_information(x, w, mi_estimator='auto', is_discrete=False)
        except AssertionError:
            fail()
        with raises(TypeError):
            mutual_information(X, X, mi_estimator='auto', is_discrete=1)

    def test_knn(self):
        try:
            mutual_information(X, X, mi_estimator='knn', n_neighbors=1)
            mutual_information(X, X, mi_estimator='knn', n_neighbors=9)
        except (KeyError, ValueError):
            fail()
        with raises(AssertionError):
            mutual_information(X, X, mi_estimator='knn', n_neighbors=0)
        with raises(AssertionError):
            mutual_information(X, X, mi_estimator='knn', n_neighbors=10)
        with raises(AssertionError):
            mutual_information(X, X, mi_estimator='knn', n_neighbors=0.1)

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
            mutual_information(x, x, mi_estimator='dr', is_discrete='auto')
            mutual_information(x, w, mi_estimator='knn', is_discrete='auto')
            mutual_information(w, w, mi_estimator='plugin', is_discrete='auto')
            mutual_information(x, x, mi_estimator='dr', is_discrete=False)
            mutual_information(x, w, mi_estimator='knn', is_discrete=False)
            mutual_information(w, w, mi_estimator='plugin', is_discrete=True)
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
        assert 0 == approx(conditional_mutual_information(z, w, w), abs=1e-6)
        assert 0 == approx(conditional_mutual_information(w, z, w), abs=1e-6)
        assert mutual_information(w, w) == approx(conditional_mutual_information(w, w, z), abs=1e-6)

    def test_mi_estimator(self):
        try:
            conditional_mutual_information(x, x, x, mi_estimator='dr', is_discrete='auto')
            conditional_mutual_information(x, x, w, mi_estimator='knn', is_discrete='auto')
            conditional_mutual_information(x, w, x, mi_estimator='knn', is_discrete='auto')
            conditional_mutual_information(x, w, w, mi_estimator='knn', is_discrete='auto')
            conditional_mutual_information(w, x, x, mi_estimator='knn', is_discrete='auto')
            conditional_mutual_information(w, x, w, mi_estimator='knn', is_discrete='auto')
            conditional_mutual_information(w, w, x, mi_estimator='knn', is_discrete='auto')
            conditional_mutual_information(w, w, w, mi_estimator='plugin', is_discrete='auto')
            conditional_mutual_information(x, x, x, mi_estimator='dr', is_discrete=False)
            conditional_mutual_information(x, x, w, mi_estimator='knn', is_discrete=False)
            conditional_mutual_information(x, w, x, mi_estimator='knn', is_discrete=False)
            conditional_mutual_information(x, w, w, mi_estimator='knn', is_discrete=False)
            conditional_mutual_information(w, x, x, mi_estimator='knn', is_discrete=False)
            conditional_mutual_information(w, x, w, mi_estimator='knn', is_discrete=False)
            conditional_mutual_information(w, w, x, mi_estimator='knn', is_discrete=False)
            conditional_mutual_information(w, w, w, mi_estimator='plugin', is_discrete=True)
        except ValueError:
            fail()
        with raises(ValueError):
            conditional_mutual_information(x, x, x, mi_estimator='')
