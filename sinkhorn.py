import torch
import numpy as np
import scipy.linalg
import scipy.stats

def sq_bw_distance(cov_hat, cov):
    # assume mean error is zero
    sqrt_cov = scipy.linalg.sqrtm(cov)
    cross_term = sqrt_cov @ cov_hat @ sqrt_cov
    cov_err = (1/2) * np.trace(cov_hat + cov - 2*scipy.linalg.sqrtm(cross_term))
    return np.real(cov_err) # singular covariances can have -0.0000... entries that yield imaginary part

def sinkhorn(cov_a, cov_b, lmbda):
    # lmbda is SCONES regularization parameter,
    #   2 sigma**2 = lambda is used in the paper
    d = len(cov_a)
    sigma = np.sqrt(lmbda/2)
    Asq = scipy.linalg.sqrtm(cov_a)
    Asqinv = np.linalg.pinv(Asq)
    Ds = scipy.linalg.sqrtm(4 * Asq @ cov_b @ Asq + sigma**4 * np.eye(d))
    Cs = (1/2) * (Asq @ Ds @ Asqinv) - (sigma**2 / 2)* np.eye(d)
    sinkhorn_cov = np.block([[cov_a, Cs], [Cs.T, cov_b]])
    return sinkhorn_cov

def sample_stats(sample):
    sample = sample.squeeze()
    sample_mean = np.mean(sample, axis=0)
    sample_cov = np.cov(sample, rowvar=False)
    return sample_mean, sample_cov


def bw_uvp(empirical_cpl, true_source_cov, true_target_cov, lmdba):
    true_cpl = sinkhorn(true_source_cov, true_target_cov, lmdba)
    var = np.real(np.trace(true_cpl))
    dist = sq_bw_distance(empirical_cpl, true_cpl) / (0.5 * var)
    return dist