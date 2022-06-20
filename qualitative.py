import numpy as np
from cpat import Compatibility, init_cpat, train_cpat
from bproj import BaryProj, init_bproj, train_bproj
from score import init_score, train_score
from scones import SCONES
import matplotlib.pyplot as plt
from score import Score
from config import Config
from datasets import Gaussian, SwissRoll
import json
import torch
import sys
import os
from sinkhorn import sample_stats, sq_bw_distance, sinkhorn, bw_uvp
from collections import defaultdict
'''
Given a configuration, train SCONES and BP and output
'''

cnf = Config("Swiss-Roll",
             source="gaussian",
             target="swiss-roll",
             l = 2,
             score_lr=0.000001,
             score_iters=1000,
             score_bs=500,
             score_noise_init=3,
             score_noise_final=0.01,
             scones_iters=1000,
             scones_bs=1000,
             device='cuda',
             score_n_classes = 10,
             score_steps_per_class = 300,
             score_sampling_lr = 0.00001,
             scones_samples_per_source=100,
             seed=2039)
torch.manual_seed(cnf.seed)
np.random.seed(cnf.seed)

cpat = init_cpat(cnf)

# If TRUE, ignore any existing pretrained models and overwrite them.
OVERWRITE = False

# Create directories for saving pretrained models if they do not already exist
touch_path = lambda p: os.makedirs(p) if not os.path.exists(p) else None
for path in ['', 'cpat', 'bproj', 'ncsn']:
    touch_path('pretrained/' + path)

# Search for and load any existing pretrained models
if ((not OVERWRITE) and os.path.exists(os.path.join("pretrained/cpat", cnf.name))):
    cpat.load(os.path.join("pretrained/cpat", cnf.name, "cpat.pt"))
else:
    train_cpat(cpat, cnf, verbose=True)

bproj = init_bproj(cpat, cnf)

if ((not OVERWRITE) and os.path.exists(os.path.join("pretrained/bproj", cnf.name))):
    bproj.load(os.path.join("pretrained/bproj", cnf.name, "bproj.pt"))
else:
    train_bproj(bproj, cnf, verbose=True)

score = init_score(cnf)

if ((not OVERWRITE) and os.path.exists(os.path.join("pretrained/score", cnf.name))):
    score.load(os.path.join("pretrained/score", cnf.name, "score.pt"))
else:
    train_score(score, cnf, verbose=True)

scones = SCONES(cpat, score, bproj, cnf)

# Sample and test the model 
n_samples = 400
Xs = cnf.source_dist.rvs(size=(n_samples,))
Xs_th = torch.FloatTensor(Xs).to(cnf.device)

bproj_Xs_th = bproj.projector(Xs_th).detach()
bproj_Xs = bproj_Xs_th.cpu().numpy()


scones_samples = scones.sample(Xs_th, verbose=False, source_init=True)

plt.subplot(1, 2, 1)
plt.scatter(*Xs.T, color="#330C2F", label="Source")
plt.scatter(*cnf.target_dist.rvs(size=(n_samples,)).T, color="#7B287D", label="Target")
plt.legend()
plt.ylim(-15, 16)
plt.xlim(-12, 16)
plt.title("Source and Target")

plt.subplot(1, 2, 2)
plt.scatter(*bproj_Xs.T, label="BPROJ", color="#7067CF")
plt.scatter(*scones_samples.reshape(-1, 2).T, label="SCONES", color="#1d3557")
plt.legend()
plt.ylim(-15, 16)
plt.xlim(-12, 16)
plt.title("Source $\\to$ Target Transportation")

plt.gcf().set_size_inches(10, 5)
plt.savefig("Source_2_Target.png")
plt.show()

#np.save("Cutout_Bproj_Gaussian->SwissRoll.npy", bproj_Xs)
#np.save("Cutout_SCONES_Gaussian->SwissRoll.npy", scones_samples)
np.save("Cutout.npy", scones_samples)
np.save("Sources.npy", Xs)
np.save("Target.npy", cnf.target_dist.rvs(size=(k,)))