from sklearn.datasets import make_moons, make_swiss_roll
import numpy as np


class TwoMoons():
    def __init__(self, noise):
        '''
        Noise is the standard deviation added to the two moons
        '''
        self.noise = noise
        self.dim = 2

    def rvs(self, size=(1,)):
        return make_moons(n_samples=np.prod(size), noise=self.noise).reshape(list(size) + [-1])

class Circle():
    def __init__(self, noise):
        '''
        Noise is the standard deviation added to the circle
        '''
        self.noise = noise
        self.dim = 2

    def rvs(self, size=(1,)):
        noise = np.random.normal(size=[np.prod(size), 2], scale=self.noise)
        circle_samples = np.random.normal(size=[size, 2])
        circle_samples = (circle_samples / np.linalg.norm(circle_samples, axis=1)[:, None]) + noise
        return circle_samples.reshape(list(size) + [-1])

class SwissRoll():
    def __init__(self, noise):
        self.noise = noise
        self.dim = 2

    def rvs(self, size):
        return make_swiss_roll(n_samples=np.prod(size), noise=self.noise)[0][:, (0, 2)].reshape(list(size) + [-1])


class Gaussian():
    def __init__(self, mean, cov):
        self.mean = np.array(mean).flatten()
        self.dim = len(self.mean)
        self.cov = np.array(cov)
        S, U = np.linalg.eigh(cov)

        S[np.isclose(S, np.zeros_like(S))] = 0
        S_pinv = np.copy(S)
        S_pinv[S_pinv != 0] = 1/S_pinv[S_pinv != 0]
        S_sqrt = np.sqrt(S)
        self.prec = U @ np.diag(S_pinv) @ U.T
        self.sqrt_cov = U @ np.diag(S_sqrt)

    def rvs(self, size):
        n_samples = np.prod(size)
        unit = np.random.normal(size=(n_samples, self.dim, 1))
        samples = self.mean.reshape((1, self.dim, 1)) + self.sqrt_cov @ unit
        return samples.reshape(list(size) + [-1])