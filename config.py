from datasets import *

def resolve_dataset(D):
    if(D == "2moon"):
        return TwoMoons(noise=0.1)
    if(D == "circle"):
        return Circle(noise=0.1)
    if(D == "swiss-roll"):
        return SwissRoll(noise=0.25)
    if(D == "gaussian"):
        return Gaussian(mean=np.array([0, 0]), cov=np.eye(2))

class Config():
    def __init__(self,
                 name,
                 source=None,
                 target=None,
                 l=10,
                 cpat_lr=0.00001,
                 cpat_iters=5000,
                 cpat_bs=500,
                 bproj_lr=0.00001,
                 bproj_iters=5000,
                 bproj_bs=500,
                 score_lr=0.00001,
                 score_iters=5000,
                 score_bs=500,
                 scones_iters=1000,
                 scones_bs=1000,
                 cov_samples=10000,
                 device='cuda',
                 score_n_classes = 20,
                 score_steps_per_class = 5,
                 score_sampling_lr = 0.001,
                 score_noise_init = 3,
                 score_noise_final = 0.01,
                 scones_samples_per_source = 10,
                 seed=2039):

        self.name = name

        self.source_dist = resolve_dataset(source)
        self.target_dist = resolve_dataset(target)
        self.source_dim = self.source_dist.dim
        self.target_dim = self.target_dist.dim

        self.l = l
        self.cpat_lr = cpat_lr
        self.cpat_iters = cpat_iters
        self.cpat_bs = cpat_bs
        self.bproj_lr = bproj_lr
        self.bproj_iters = bproj_iters
        self.bproj_bs = bproj_bs
        self.score_lr = score_lr
        self.score_iters = score_iters
        self.score_bs = score_bs
        self.score_n_classes = score_n_classes
        self.score_steps_per_class = score_steps_per_class
        self.score_sampling_lr = score_sampling_lr
        self.score_noise_init = score_noise_init
        self.score_noise_final = score_noise_final
        self.scones_iters = scones_iters
        self.scones_samples_per_source = scones_samples_per_source
        self.scones_bs = scones_bs
        self.cov_samples = cov_samples
        self.device = device
        self.seed = seed

class GaussianConfig():
    def __init__(self,
                 name,
                 source_cov=None,
                 target_cov=None,
                 scale_huh=False,
                 l=10,
                 cpat_lr=0.00001,
                 cpat_iters=5000,
                 cpat_bs=500,
                 bproj_lr=0.00001,
                 bproj_iters=5000,
                 bproj_bs=500,
                 scones_sampling_lr=0.01,
                 scones_iters=1000,
                 scones_bs=1000,
                 cov_samples=10000,
                 device='cuda',
                 score_n_classes = 20,
                 score_steps_per_class = 5,
                 score_sampling_lr = 0.001,
                 seed=2039):

        self.name = name
        self.source_cov = np.load(source_cov)
        self.source_dim = len(self.source_cov)
        self.target_cov = np.load(target_cov)
        self.target_dim = len(self.target_cov)
        if(scale_huh):
            d = len(self.source_cov)
            self.source_cov = self.source_cov / d
            self.target_cov = self.target_cov / d
        self.source_dist = Gaussian(mean=np.zeros(self.source_dim), cov=self.source_cov)
        self.target_dist = Gaussian(mean=np.zeros(self.target_dim), cov=self.target_cov)
        self.l = l
        self.cpat_lr = cpat_lr
        self.cpat_iters = cpat_iters
        self.cpat_bs = cpat_bs
        self.bproj_lr = bproj_lr
        self.bproj_iters = bproj_iters
        self.bproj_bs = bproj_bs
        self.score_n_classes = score_n_classes
        self.score_steps_per_class = score_steps_per_class
        self.score_sampling_lr = score_sampling_lr
        self.scones_sampling_lr = scones_sampling_lr
        self.scones_iters = scones_iters
        self.scones_bs = scones_bs
        self.cov_samples = cov_samples
        self.device = device
        self.seed = seed