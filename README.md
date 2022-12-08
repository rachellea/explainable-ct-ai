## Description

This repository contains Python code to train and evaluate convolutional neural network models (CNNs)
on the task of multiple abnormality prediction from whole chest CT volumes.

This repository includes code for numerous projects and models, including code related to the following research works:

Draelos, R. L., & Carin, L. "Explainable multiple abnormality classification of chest CT volumes." *Artificial Intelligence in Medicine* (2022).
* AxialNet
* HiResCAM

Draelos, R.L. "Towards fully automated interpretation of volumetric medical images with deep learning." Duke University PhD Thesis (2022).

## Usage

Currently this repository represents the final state of my primary PhD codebase
after I defended and graduated. The `runs` directory includes "run files" that,
when moved to the root directory, can be run as

`python runfile.py`

I apologize that some of these run files assume an earlier state of the repo
where classes/functions had slightly different interfaces. At some point I hope
to "tutorialize" this repo so that it's straightforward to run everything
of interest, but for now I figure sharing some code is better than sharing no
code, so here you go :)

## Requirements

The requirements are specified in *ct_environment.yml* and include
PyTorch, numpy, pandas, sklearn, scipy, and matplotlib.

To create the conda environment run:

`conda env create -f ct_environment.yml`

The code can also be run using the Singularity container defined [in this repository](https://github.com/rachellea/research-container).

## Unit Testing

This repo uses the Python unittest module for unit testing. You can use
unittest discover to run the unit tests.

## Dataset

This research code was developed using the RAD-ChestCT dataset. The models
in this codebase can be trained on the RAD-ChestCT dataset. CT scans from RAD-ChestCT
are publicly available on Zenodo [at this link](https://zenodo.org/record/6406114).