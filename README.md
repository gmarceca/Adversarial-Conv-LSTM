# Adversarial Conv-LSTM

(Work in progress)

Implementation of an adversarial Conv-LSTM model for classification of plasma confinement states 
and cross-machine domain adaptation

GradientReversal layer implementation based on the following code
https://github.com/ajgallego/DANN

## Results

<img src="https://github.com/gmarceca/Adversarial-Conv-LSTM/blob/main/DANN_model_kappa_vs_epochs_TCV2AUG_adv.png" width="400" height="400" />

## Installation

<b># Installation of Miniconda from scratch</b>
- Get and install Miniconda:
    1. `cd your_project/` (Miniconda packages might require a significant space ~Gbs)
    1. `wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`
    2. `bash Miniconda3-latest-Linux-x86_64.sh`
    3. `export PATH="/home/user/your_project/miniconda3/bin:$PATH"` (or where you have decided to install miniconda3)

<b># Create an environment</b>
- An environment file is provided for version compatibility.\
`conda create --name my_gpu_env --file tf1p12_env.yml`\
*This version of tensorflow works with cuda-9.0 as highlighted here*
https://www.tensorflow.org/install/source#gpu

## Preparation of Experiments
(Work in progress for a more complete documentation)
### Run an experiment
`source run_experiment.sh`
