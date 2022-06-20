import torch
from nets import FCNN, FCNN2, FCCritic
import os
import shutil
import numpy as np
import tqdm

class Compatibility():
    def __init__(self, phi, psi, cnf):
        self.phi = phi
        self.psi = psi
        self.cnf = cnf

    def save(self, path, train_idx=None):
        if(os.path.exists(path)):
            shutil.rmtree(path)
        os.makedirs(path)
        states = {"phi": self.phi.state_dict(), "psi": self.psi.state_dict()}
        if(train_idx is None):
            torch.save(states, os.path.join(path, "cpat.pt"))
        else:
            torch.save(states, os.path.join(path, f"cpat_{train_idx}.pt"))

    def load(self, path):
        cpat_dict = torch.load(path)
        self.phi.load_state_dict(cpat_dict["phi"])
        self.psi.load_state_dict(cpat_dict["psi"])

    def violation(self, x, y):
        return self.phi(x) + self.psi(y) - torch.sum((x - y)**2, dim=1).view((-1, 1))

    def penalty(self, x, y):
        l = self.cnf.l
        return l * (torch.exp((1/l) * self.violation(x, y)) - 1)

    def dual_obj(self, x, y):
        return torch.mean(self.phi(x) + self.psi(y) - self.penalty(x, y))

    def density(self, x, y):
        l = self.cnf.l
        return torch.exp((1 / l) * self.violation(x, y)).view((-1, 1,))

    def covariance(self, x, y):
        # Given batches x and y, estimate the covariance of the joint distribution on x, y as weighted by
        #   these compatibilities
        dim = np.prod(x.shape[1:])
        density = self.density(x, y).view((-1, 1, 1))
        joint = torch.cat((x, y), axis=1).view((-1, 2*dim, 1))
        sample_corr = joint @ joint.transpose(1, 2)
        est_corr = torch.mean(density * sample_corr, axis=0)
        return est_corr.detach().cpu().numpy()

    def score(self, source, target):
        # compute grad log cpat(source, target) wrt target
        temp = 1 / self.cnf.l
        grad_psi = torch.cat(torch.autograd.grad(outputs=list(self.psi(target)), inputs=[target]), dim=1)
        transport_grad = temp*(grad_psi - 2*(target - source))
        return transport_grad

def init_cpat(cnf):
    phi = FCCritic(input_dim=cnf.source_dist.dim, hidden_layer_dims=[4096, 4096, 4096]).to(cnf.device)
    psi = FCCritic(input_dim=cnf.target_dist.dim, hidden_layer_dims=[4096, 4096, 4096]).to(cnf.device)
    return Compatibility(phi, psi, cnf)

def train_cpat(cpat, cnf, verbose=True):
    bs = cnf.cpat_bs
    lr = cnf.cpat_lr
    iters = cnf.cpat_iters

    source_dist = cnf.source_dist
    target_dist = cnf.target_dist

    opt = torch.optim.Adam(params=list(cpat.phi.parameters()) + list(cpat.psi.parameters()), lr=lr, betas=(0.9, 0.999))

    if(verbose):
        t = tqdm.tqdm(total=iters, desc='', position=0)
    for i in range(iters):
        source_sample = torch.FloatTensor(source_dist.rvs(size=(bs,))).to(cnf.device)
        target_sample = torch.FloatTensor(target_dist.rvs(size=(bs,))).to(cnf.device)

        opt.zero_grad()
        obj = cpat.dual_obj(source_sample, target_sample)
        (-obj).backward()
        opt.step()
        avg_density = torch.mean(cpat.density(source_sample, target_sample))

        if(verbose):
            t.set_description(f"Objective: {round(obj.item(), 5)} - Average Density: {round(avg_density.item(), 5)}")
            t.update(1)

        if(i % 500 == 0):
            cpat.save(os.path.join("pretrained/cpat", cnf.name), train_idx=i)
            cpat.save(os.path.join("pretrained/cpat", cnf.name))
