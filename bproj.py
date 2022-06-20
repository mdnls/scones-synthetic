import torch
from nets import FCNN, FCNN2
import os
import shutil
import numpy as np
import tqdm

class BaryProj():
    def __init__(self, cpat, projector, cnf):
        self.cpat = cpat
        self.projector = projector
        self.cnf = cnf

    def save(self, path, train_idx=None):
        if(os.path.exists(path)):
            shutil.rmtree(path)
        os.makedirs(path)
        state = self.projector.state_dict()
        if(train_idx is None):
            torch.save(state, os.path.join(path, "bproj.pt"))
        else:
            torch.save(state, os.path.join(path, f"bproj_{train_idx}.pt"))

    def load(self, path):
        self.projector.load_state_dict(torch.load(path))

    def mapping_error(self, x, y):
        err = torch.sum((self.projector(x) - y)**2, dim=1).view((-1, 1))
        density = self.cpat.density(x, y).view((-1, 1))
        return torch.mean(err * density)

    def covariance(self, source_samples):
        dim = np.prod(source_samples.shape[1:])
        transport = self.projector(source_samples).view((-1, dim, 1))
        source_samples = source_samples.view((-1, dim, 1))
        joint = torch.cat((source_samples, transport), axis=1).view((-1, 2*dim))
        return np.cov(joint.detach().cpu().numpy(), rowvar=False)

def init_bproj(cpat, cnf):
    ds = cnf.source_dim
    dt = cnf.target_dim
    T = FCNN(dims=[ds, 2048, 2048, 2048, dt], batchnorm=True).to(cnf.device)
    return BaryProj(cpat, T, cnf)


def train_bproj(bproj, cnf, verbose=True):
    bs = cnf.bproj_bs
    lr = cnf.bproj_lr
    iters = cnf.bproj_iters

    source_dist = cnf.source_dist
    target_dist = cnf.target_dist

    opt = torch.optim.Adam(params=bproj.projector.parameters(), lr=lr)

    if(verbose):
        t = tqdm.tqdm(total=iters, desc='', position=0)
    for i in range(iters):
        source_sample = torch.FloatTensor(source_dist.rvs(size=(bs,))).to(cnf.device)
        target_sample = torch.FloatTensor(target_dist.rvs(size=(bs,))).to(cnf.device)

        opt.zero_grad()
        obj = bproj.mapping_error(source_sample, target_sample)
        obj.backward()
        opt.step()

        if(verbose):
            t.set_description("Objective: {:.2E}".format(obj.item()))
            t.update(1)

            if(i % 1000 == 0):
                print("\nCovariance:")
                source_sample = torch.FloatTensor(source_dist.rvs(size=(10000,))).to(cnf.device)
                print(bproj.covariance(source_sample))

        if(i % 500 == 0):
            bproj.save(os.path.join("pretrained/bproj", cnf.name), train_idx=i)
            bproj.save(os.path.join("pretrained/bproj", cnf.name))
