import torch
from nets import FCNN, FCNN2, FCCritic
import os
import shutil
from tqdm import trange
import numpy as np
import tqdm
from config import Config, GaussianConfig
from score import Score

class GaussianSCONES():
    def __init__(self, cpat, prior, bproj, cnf):
        self.cpat = cpat
        self.bproj = bproj
        self.prior = prior
        self.cnf = cnf
        self.tgt_prec = torch.FloatTensor(np.linalg.inv(cnf.target_cov)).to(cnf.device)

    def score(self, source, target, s=1):
        # compute grad log p wrt targets
        cpat_grad = self.cpat.score(source, target)
        prior_grad = self.prior.score(target, s=s)
        return cpat_grad + prior_grad

    def sample(self, source, verbose=True):
        bs = self.cnf.scones_bs
        eps = self.cnf.scones_sampling_lr
        n_batches = int(np.ceil(self.cnf.cov_samples / bs))
        source_batches = [source[bs * i: bs * (i + 1)] for i in range(n_batches)]
        target_batches = [self.bproj.projector(s) for s in source_batches]
        samples = []

        for b in range(n_batches):
            source = source_batches[b]
            target = target_batches[b]
            for i in range(self.cnf.scones_iters):
                Z = torch.randn_like(target)
                score = self.score(source, target)
                with torch.no_grad():
                    target = target + (eps / 2) * score + np.sqrt(eps) * Z
                target.requires_grad = True
                if (verbose and i % 100 == 0):
                    cov = self._est_covariance(source, target)
                    print("")
                    print(cov)
            samples.append(target)
        return torch.cat(samples, dim=0)

    def _est_covariance(self, source, target):
        source = source.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        joint = np.concatenate((source, target), axis=1).reshape((len(source), -1))
        return np.cov(joint, rowvar=False)

    def covariance(self, source, verbose=True):
        samples = self.sample(source, verbose=verbose)
        joint = np.concatenate((source.detach().cpu().numpy(), samples.detach().cpu().numpy()), axis=1)
        return np.cov(joint, rowvar=False)

class SCONES():
    def __init__(self, cpat, score, bproj, cnf):
        self.cpat = cpat
        self.bproj = bproj
        self.score_est = score
        self.cnf = cnf

    def score(self, source, target, noise_std=1):
        # compute grad log p wrt targets
        cpat_grad = self.cpat.score(source, target)
        prior_grad = self.score_est.score(target, noise_std=noise_std)
        return cpat_grad + prior_grad

    def sample(self, source, source_init=False, verbose=True):
        n_samples = self.cnf.scones_samples_per_source * len(source)

        Xs = torch.stack([source] * self.cnf.scones_samples_per_source, dim=1).view(n_samples, -1).to(self.cnf.device)
        if(source_init):
            Xt = torch.clone(Xs)
        else:
            Xt = torch.randn(size=[n_samples, 2]).to(self.cnf.device)

        for s in self.score_est.noise_scales:
            for _ in range(self.score_est.steps_per_class):
                a = self.score_est.sampling_lr * (s / self.score_est.noise_scales[-1])**2
                noise = torch.randn(size=[n_samples, 2]).to(self.cnf.device)
                Xt.requires_grad = True
                scr = self.score(Xs, Xt, s)
                with torch.no_grad():
                    Xt = Xt + a * scr + np.sqrt(2*a) * noise
        # denoise via tweedie's identity
        Xt.requires_grad = True
        Xt = Xt + self.score_est.noise_scales[-1]**2 * self.score(Xs, Xt, self.score_est.noise_scales[-1])
        return Xt.detach().cpu().numpy().reshape(len(source), self.cnf.scones_samples_per_source, -1)
