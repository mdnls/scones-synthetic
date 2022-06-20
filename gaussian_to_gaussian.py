import numpy as np
from cpat import Compatibility, init_cpat, train_cpat
from bproj import BaryProj, init_bproj, train_bproj
from scones import GaussianSCONES
from score import GaussianScore
from config import GaussianConfig
from datasets import Gaussian
import scipy.stats
import json
import torch
import sys
from tqdm import trange
import os
from sinkhorn import sample_stats, sq_bw_distance, sinkhorn, bw_uvp
from collections import defaultdict


if __name__ == "__main__":
    OVERWRITE = True
    dims = [2, 16, 64, 128, 256]
    results = { str(d): {"runs": []} for d in dims }
    for d in dims[::-1]:
        for i in trange(3):
            cnf = GaussianConfig(name=f"l={2 * d}_d={d}_k=0",
                   source_cov=f"data/d={d}/{i}/source_cov.npy",
                   target_cov=f"data/d={d}/{i}/target_cov.npy",
                   scale_huh=False,
                   cpat_bs=500,
                   cpat_iters=5000,
                   cpat_lr=0.000001 * d,
                   bproj_bs=500,
                   bproj_iters=5000,
                   bproj_lr=0.00001,
                   scones_iters=1000,
                   device='cpu',
                   l=d * 2)

            torch.manual_seed(cnf.seed)
            np.random.seed(cnf.seed)

            cpat = init_cpat(cnf)

            if ((not OVERWRITE) and os.path.exists(os.path.join("pretrained/cpat", cnf.name))):
                cpat.load(os.path.join("pretrained/cpat", cnf.name, "cpat.pt"))
            else:
                train_cpat(cpat, cnf, verbose=True)

            bproj = init_bproj(cpat, cnf)

            if ((not OVERWRITE) and os.path.exists(os.path.join("pretrained/bproj", cnf.name))):
                bproj.load(os.path.join("pretrained/bproj", cnf.name, "bproj.pt"))
            else:
                train_bproj(bproj, cnf, verbose=True)

            prior = GaussianScore(cnf.target_dist, cnf)
            scones = GaussianSCONES(cpat, prior, bproj, cnf)

            Xs = cnf.source_dist.rvs(size=(cnf.cov_samples,))
            Xs_th = torch.FloatTensor(Xs).to(cnf.device)

            mean = np.zeros((cnf.source_dim + cnf.target_dim,))

            bproj_cov = bproj.covariance(Xs_th)
            scones_cov = scones.covariance(Xs_th, verbose=False)

            bproj_bw_uvp = bw_uvp(bproj_cov, cnf.source_cov, cnf.target_cov, cnf.l)
            scones_bw_uvp = bw_uvp(scones_cov, cnf.source_cov, cnf.target_cov, cnf.l)

            results[str(d)]['runs'].append({"d": d, "bproj-bw-uvp": bproj_bw_uvp, "scones-bw-uvp": scones_bw_uvp})
        bproj_avg_bw_uvp = np.mean([run['bproj-bw-uvp'] for run in results[str(d)]['runs']])
        scones_avg_bw_uvp = np.mean([run['scones-bw-uvp'] for run in results[str(d)]['runs']])
        print(f"BPROJ average BW-UVP at d={d}: {bproj_avg_bw_uvp}")
        print(f"SCONES average BW-UVP at d={d}: {scones_avg_bw_uvp}")
        results[str(d)]['bproj-mean-bw-uvp'] = bproj_avg_bw_uvp
        results[str(d)]['scones-mean-bw-uvp'] = scones_avg_bw_uvp
    with open(f"Results_{'_'.join([str(x) for x in dims])}.json", "w+") as f_out:
        f_out.write(json.dumps(results))


