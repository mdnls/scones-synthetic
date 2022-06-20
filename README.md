# SCONES: Synthetic Experiments

**Score-based Generative Neural Networks for Large-Scale Optimal Transport**. ([on ArXiv](https://arxiv.org/abs/2110.03237)) <br />
_Max Daniels, Tyler Maunu, Paul Hand_. 

This repository contains code for running synthetic experiments, including those used to generate Table 2 (Gaussian to Gaussian BW-UVP scores) and Figure 4 (Gaussian to Swiss Roll transportation in 2D). A sister repository, used to run all large-scale experiments, can be found [here](https://github.com/mdnls/scones)

## Setup
The required packages can be found in `requirements.txt`. To create a new conda environment with these packages installed, use 

`conda create --name <env> --file requirements.txt`.

## Running the code 
The two main entry points are `gaussian_to_gaussian.py` and `qualitative.py`. These files are preconfigured and can be run out-of-the-box, or one can set up custom experiments by changing the configuration inputs, which may be found in 

**BW-UVP Experiments**: run `python gaussian_to_gaussian.py` to recreate Table 2, which compares our sampling algorithm to the ground truth in BW-UVP distance. 

The code uses instances of `GaussianConfig` (in `config.py`) to configure experiments. To run customized experiments, such as changing the dimensionality or sampling parameters, pass a customized instance of the config object as in the example above. 

**Swiss-roll to Gaussian**: run `python qualitative.py` to recreate Figure 4, which simulates our sampling algorithm for a synthetic transportation task between low-dimensional synthetic datasets. 

This code uses instances of `Config` (in `config.py`) to configure experiments. To run customized experiments, such as changing the source and target datasets or the regularization parameters, pass a customized instance of the configuration object as above. 

_Note_: sampling requires training from scratch a SCONES model for the target distribution. The SCONES model has multiple components: the score-based generative model, the compatibility function, and (optionally) a barycentric projector. Pretrained models for transport to the Swiss Roll dataset, which were used to generate Figure 4, can be found [here](https://drive.google.com/drive/folders/1MOgKe-ispWehFLWlHXhvxxawQF1WIeHp?usp=sharing) (merge with the existing `pretrained/` directory).
