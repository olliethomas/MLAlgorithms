#from mla.base import BaseEstimator
#from mla.svm.kernerls import Linear
import numpy as np
import logging

np.random.seed(9999)

"""
References:
The Simplified SMO Algorithm http://cs229.stanford.edu/materials/smo.pdf
"""

class Linear(object):
    def __call__(self, x, y):
        return np.dot(x, y.T)
    def __repr__(self):
        return 'Linear kernel'

class BaseEstimator(object):
    X = None
    y = None
    y_required = True

    def _setup_input(self, X, y=None):
        """Ensure inputs to an estimator are in the expected format.
        Ensures X and y are stored as numpy ndarrays by converting from an
        array-like object if necessary. Enables estimators to define whether
        they require a set of y target values or not with y_required, e.g.
        kmeans clustering requires no target labels and is fit against only X.
        Parameters
        ----------
        X : array-like
            Feature dataset.
        y : array-like
            Target values. By default is required, but if y_required = false
            then may be omitted.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError('Number of features must be > 0')

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError('Missed required argument y')

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError('Number of features must be > 0')

        self.y = y

    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        if self.X is not None:
            return self._predict(X)
        else:
            raise ValueError('You must call `fit` before `predict`')

    def _predict(self, X=None):
        raise NotImplementedError()


class SVM(BaseEstimator):
    def __init__(self, C=1.0, kernel=None, tol=1e-3, max_iter=100):
        """Support vector machines implementation using simplified SMO optimization.

        Parameters
        ----------
        C : float, default 1.0
        kernel : Kernel object
        tol : float , default 1e-3
        max_iter : int, default 100
        """
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        if kernel is None:
            self.kernel = Linear()
        else:
            self.kernel = kernel

        self.b = 0
        self.alpha = None
        self.K = None

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.K = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            self.K[:, i] = self.kernel(self.X, self.X[i, :])
        self.alpha = np.zeros(self.n_samples)
        self.sv_idx = np.arange(0, self.n_samples)
        return self._train()

    def _train(self):
        iters = 0
        while iters < self.max_iter:
            iters += 1
            alpha_prev = np.copy(self.alpha)

            for j in range(self.n_samples):
                # Pick random i
                i = self.random_index(j)

                eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                print("i = ", i, "j = ", j)
                print("eta = ", eta, "self.K[i, j] then ii and ij = ", self.K[i, j], self.K[i, i], self.K[j, j])
                if eta >= 0:
                    continue
                L, H = self._find_bounds(i, j)

                # Error for current examples
                e_i, e_j = self._error(i), self._error(j)

                # Save old alphas
                alpha_io, alpha_jo = self.alpha[i], self.alpha[j]

                # Update alpha
                self.alpha[j] -= (self.y[j] * (e_i - e_j)) / eta
                self.alpha[j] = self.clip(self.alpha[j], H, L)

                self.alpha[i] = self.alpha[i] + self.y[i] * self.y[j] * (alpha_jo - self.alpha[j])

                # Find intercept
                print("b = ", self.b, " self.y[i] = ", self.y[i], "alpha[i]", self.alpha[i])
                b1 = self.b - e_i - self.y[i] * (self.alpha[i] - alpha_jo) * self.K[i, i] - \
                     self.y[j] * (self.alpha[j] - alpha_jo) * self.K[i, j]
                print("b1 = ", b1)
                b2 = self.b - e_j - self.y[j] * (self.alpha[j] - alpha_jo) * self.K[j, j] - \
                     self.y[i] * (self.alpha[i] - alpha_io) * self.K[i, j]
                print("b2 = ", b2)
                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = 0.5 * (b1 + b2)
                print("self.b = ", self.b)

            # Check convergence
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break
        logging.info('Convergence has reached after %s.' % iters)

        # Save support vectors index
        self.sv_idx = np.where(self.alpha > 0)[0]
        print("w = ", self.w)
        print("b = ", self.b)
        print("alphas = ", self.alpha)

    def _predict(self, X=None):
        n = X.shape[0]
        result = np.zeros(n)
        for i in range(n):
            result[i] = np.sign(self._predict_row(X[i, :]))
        return result

    def _predict_row(self, X):
        self.w = np.dot((self.alpha * self.y), self.X)
        k_v = self.kernel(self.X[self.sv_idx], X)
        return np.dot((self.alpha[self.sv_idx] * self.y[self.sv_idx]).T, k_v.T) + self.b

    def clip(self, alpha, H, L):
        if alpha > H:
            alpha = H
        if alpha < L:
            alpha = L
        return alpha

    def _error(self, i):
        """Error for single example."""
        return self._predict_row(self.X[i]) - self.y[i]

    def _find_bounds(self, i, j):
        """Find bounds L and H such that L ≤ alpha ≤ H.
        alpha must satisfy the constraint 0 ≤ αlpha ≤ C.
        """
        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C - self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        return L, H

    def random_index(self, z):
        i = z
        while i == z:
            i = np.random.randint(0, self.n_samples - 1)
        return i

s = SVM()
s.fit([[-1,-4],[-2,-3],[-1,-1],[-3,-2],[2,1],[1,1],[1,2]], [-1,-1,-1,-1,1,1,1])
x = np.array([[-2,-2]])
print(s.predict(x), s.alpha, s.b)
print(s.w)