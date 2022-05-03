# Improved Adversarial Robustness via Abstract Interpretation

*Authors: Zachary DeStefano, Ildebrando Magnani, William Merrill*

Final project for [Foundations of Machine Learning](https://cs.nyu.edu/~mohri/mlsp22/) at NYU with Professor Mehryar Mohri.

This codebase is based on a clone of [DiffAI](https://github.com/eth-sri/diffai), a toolkit for applying abstract interpretation to neural networks.

We provide documentation of installation and commands used to run experiments.

## Environment Setup

After cloning this repository, run the following from the `diffai/` directory:

```shell
# Create conda environment. Need to install conda or miniconda first.
conda create -n diffai python=3.6
conda activate diffai # Will need to do this every time you want to use this version of Python.

# Install legacy versions no longer supported by pip.
conda install pytorch=0.4.1 torchvision cuda90 -c pytorch
conda install torchvision=0.2.1 cuda90

# Install the rest of the dependencies with pip.
pip install numpy six future forbiddenfruit matplotlib
```

Every time you run code in the repo, you should:
```shell
conda activate diffai
```

## Certification

First, download the [`convSmall`](https://www.dropbox.com/sh/66obogmvih79e3k/AACfzqaT7kwf44Ksh1bVhUb1a/basic_nets/CIFAR10/width_2_255/ConvSmall__LinMix_a_IFGSM_w_Lin_00.0110020__k_3__b_InSamp_Lin_0115050__w_Lin_00.0115050___bw_Lin_00.515050___checkpoint_301_with_0.561.pynet?dl=0) trained model from the original DIFFAI repo.

```shell
python . -d Point --epochs 1 --dont-write --test-freq 1 -t Box(0.01) --test out/convSmall.pynet -D=CIFAR10
```

Links to other trained models can also be found in README-ORIGINAL.md, and command line arguments can be modified appropriately to work with them. You can also of course slot in your own trained models.

All certification experiments can be replicated by running trials.sh for `radius=.01` and `radius=.02`:

```shell
radius=.01 source trials.sh
radius=.02 source trials.sh
```

To then generate plots:
```shell
python plot.py
```

## Training

The command line flags to replicate experiments with abstract interpretation during training are:

```shell
python . -d="LinMix(a=Point(), b=Box(w=Lin(0,0.015686,100,0)), bw=Lin(0,1,100,0))" --batch-size 50 --epochs 101 --dont-write --test-freq 5 -t Box --width 0.031373 -D=CIFAR10 -k=4 -n=ConvSmall | tee mylog/ls-4-255.txt
python . -d="LinMix(a=Point(), b=Box(w=Lin(0,0.031373,100,0)), bw=Lin(0,1,100,0))" --batch-size 50 --epochs 101 --dont-write --test-freq 5 -t Box --width 0.031373 -D=CIFAR10 -k=4 -n=ConvSmall | tee mylog/ls-8-255.txt
python . -d="LinMix(a=Point(), b=Box(w=Lin(0,0.047059,100,0)), bw=Lin(0,1,100,0))" --batch-size 50 --epochs 101 --dont-write --test-freq 5 -t Box --width 0.031373 -D=CIFAR10 -k=4 -n=ConvSmall | tee mylog/ls-12-255.txt
python . -d="LinMix(a=Point(), b=Box(w=Lin(0,0.062745,100,0)), bw=Lin(0,1,100,0))" --batch-size 50 --epochs 101 --dont-write --test-freq 5 -t Box --width 0.031373 -D=CIFAR10 -k=4 -n=ConvSmall | tee mylog/ls-16-255.txt

python . -d="LinMix(a=Point(), b=Box(w=Lin(0,0.015686,100,0)), bw=Lin(0,1,100,0))" --batch-size 50 --epochs 101 --dont-write --test-freq 5 -t Box --width 0.031373 -D=CIFAR10 -k=4 -n=ConvMed | tee mylog/lm-4-255.txt
python . -d="LinMix(a=Point(), b=Box(w=Lin(0,0.031373,100,0)), bw=Lin(0,1,100,0))" --batch-size 50 --epochs 101 --dont-write --test-freq 5 -t Box --width 0.031373 -D=CIFAR10 -k=4 -n=ConvMed | tee mylog/lm-8-255.txt
python . -d="LinMix(a=Point(), b=Box(w=Lin(0,0.047059,100,0)), bw=Lin(0,1,100,0))" --batch-size 50 --epochs 101 --dont-write --test-freq 5 -t Box --width 0.031373 -D=CIFAR10 -k=4 -n=ConvMed | tee mylog/lm-12-255.txt
python . -d="LinMix(a=Point(), b=Box(w=Lin(0,0.062745,100,0)), bw=Lin(0,1,100,0))" --batch-size 50 --epochs 101 --dont-write --test-freq 5 -t Box --width 0.031373 -D=CIFAR10 -k=4 -n=ConvMed | tee mylog/lm-16-255.txt

python . -d="LinMix(a=Point(), b=Box(w=Lin(0,0.015686,100,0)), bw=Lin(0,1,100,0))" --batch-size 50 --epochs 101 --dont-write --test-freq 5 -t Box --width 0.031373 -D=CIFAR10 -k=4 -n=ConvBig | tee mylog/lb-4-255.txt
python . -d="LinMix(a=Point(), b=Box(w=Lin(0,0.031373,100,0)), bw=Lin(0,1,100,0))" --batch-size 50 --epochs 101 --dont-write --test-freq 5 -t Box --width 0.031373 -D=CIFAR10 -k=4 -n=ConvBig | tee mylog/lb-8-255.txt
python . -d="LinMix(a=Point(), b=Box(w=Lin(0,0.047059,100,0)), bw=Lin(0,1,100,0))" --batch-size 50 --epochs 101 --dont-write --test-freq 5 -t Box --width 0.031373 -D=CIFAR10 -k=4 -n=ConvBig | tee mylog/lb-12-255.txt
python . -d="LinMix(a=Point(), b=Box(w=Lin(0,0.062745,100,0)), bw=Lin(0,1,100,0))" --batch-size 50 --epochs 101 --dont-write --test-freq 5 -t Box --width 0.031373 -D=CIFAR10 -k=4 -n=ConvBig | tee mylog/lb-16-255.txt
```

You can then use clean.py to extract well-formatted data from the generated logs.
