import torch
from nets import FCNN, FCNN2
import os
import shutil
import numpy as np
import tqdm
from config import Config
import matplotlib.pyplot as plt

class Score():
    def __init__(self, score_net, cnf, posterior_score=None):
        self.score_net = score_net
        self.cnf = cnf
        self.noise_scales = np.geomspace(start=cnf.score_noise_init, stop=cnf.score_noise_final, num=cnf.score_n_classes)
        self.num_classes = cnf.score_n_classes
        self.steps_per_class = cnf.score_steps_per_class
        self.sampling_lr = cnf.score_sampling_lr
        self.posterior_score = posterior_score
        self._noise_scales_th = torch.FloatTensor(self.noise_scales).to(cnf.device)

    def save(self, path, train_idx=None):
        if(os.path.exists(path)):
            shutil.rmtree(path)
        os.makedirs(path)
        state = self.score_net.state_dict()
        if(train_idx is None):
            torch.save(state, os.path.join(path, "score.pt"))
        else:
            torch.save(state, os.path.join(path, f"score_{train_idx}.pt"))

    def load(self, path):
        self.score_net.load_state_dict(torch.load(path))

    def score(self, x, noise_std=1):
        if(self.posterior_score is not None):
            return self.score_net(x) / noise_std + self.posterior_score(x)
        else:
            return self.score_net(x) / noise_std

    def dsm_loss(self, sample):
        noise = torch.randn([self.num_classes] + list(sample.size())).to(self.cnf.device)
        perturbed_samples = noise * self._noise_scales_th.reshape([-1, 1, 1]) + torch.stack([sample] * self.num_classes, dim = 0)
        d = self.cnf.target_dist.dim
        obj = (d/2) * torch.mean((self.score_net(perturbed_samples.view([-1, d])) + noise.view([-1, d]))**2)
        return obj

    def sample(self, size, Xs=None):
        if(Xs is None):
            Xs = torch.randn(size=[np.prod(size), 2]).to(self.cnf.device)
        with torch.no_grad():
            for s in self.noise_scales:
                for _ in range(self.steps_per_class):
                    a = self.sampling_lr * (s / self.noise_scales[-1])**2
                    noise = torch.randn(size=[np.prod(size), 2]).to(self.cnf.device)
                    Xs = Xs + a * self.score(Xs, s) + np.sqrt(2*a) * noise
        # denoise via tweedie's identity
        Xs = Xs + self.noise_scales[-1]**2 * self.score(Xs, self.noise_scales[-1])
        return Xs.reshape(list(size) + [-1])

def init_score(cnf):
    d = cnf.target_dim
    T = FCNN(dims=[d, 2048, 2048, 2048, 2048, d], batchnorm=True).to(cnf.device)
    return Score(T, cnf)

def train_score(score, cnf, verbose=True):
    bs = cnf.score_bs
    lr = cnf.score_lr
    iters = cnf.score_iters

    target_dist = cnf.target_dist

    opt = torch.optim.Adam(params=score.score_net.parameters(), lr=lr)

    if(verbose):
        t = tqdm.tqdm(total=iters, desc='', position=0)
    for i in range(iters):
        target_sample = torch.FloatTensor(target_dist.rvs(size=(bs,))).to(cnf.device)

        opt.zero_grad()
        obj = score.dsm_loss(target_sample)
        obj.backward()
        opt.step()

        if(verbose):
            t.set_description("Objective: {:.2E}".format(obj.item()))
            t.update(1)

        if(i % 500 == 0):
            score.save(os.path.join("pretrained/score", cnf.name), train_idx=i)
            score.save(os.path.join("pretrained/score", cnf.name))

class GaussianScore():
    def __init__(self, gaussian, cnf):
        self.gaussian = gaussian
        self.cnf = cnf
        self.prec = torch.FloatTensor(gaussian.prec).to(cnf.device)
        self.dim = gaussian.dim

    def score(self, x, s=0):
        if(s > 0):
            prec = np.linalg.inv(self.gaussian.cov + s * np.eye(self.dim))
        else:
            prec = self.prec
        score = (-prec @ x.view((-1, self.dim, 1)))
        return score.view((-1, self.dim))

if __name__ == "__main__":
    cnf = Config("Swiss-Roll",
                 source="gaussian",
                 target="swiss-roll",
                 score_lr=0.000001,
                 score_iters=1000,
                 score_bs=500,
                 score_noise_init=3,
                 score_noise_final=0.01,
                 scones_iters=1000,
                 scones_bs=1000,
                 device='cuda',
                 score_n_classes = 10,
                 score_steps_per_class = 10,
                 score_sampling_lr = 0.0001,
                 seed=2039)
    ex_samples = cnf.target_dist.rvs(size=(1000,))
    score = init_score(cnf)
    #train_score(score, cnf, verbose=True)
    score.load(os.path.join("score", cnf.name, "score.pt"))
    learned_samples = score.sample(size=(1000,)).detach().cpu().numpy()
    plt.subplot(1, 2, 1)
    plt.scatter(*ex_samples.T)
    plt.subplot(1, 2, 2)
    plt.scatter(*learned_samples.T)
    plt.show()
