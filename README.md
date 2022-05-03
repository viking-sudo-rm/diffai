# Improved Adversarial Robustness via Abstract Interpretation

*Authors: Zachary DeStefano, Ildebrando Magnani, William Merrill*

Final projects for [Foundations of Machine Learning](https://cs.nyu.edu/~mohri/mlsp22/) at NYU with Professor Mehryar Mohri.

This codebase is based on a clone of DiffAI, a toolkit for applying abstract interpretation to neural networks.

We provide documentation of installation and commands used to run experiments.

## Installation

```shell
# Clone repo
git clone https://github.com/eth-sri/diffai
cd diffai

# Create conda environment. Need to install conda or miniconda first.
conda create -n diffai python=3.6
conda activate diffai # Will need to do this every time you want to use this version of Python.

# Install legacy versions no longer supported by pip.
conda install pytorch=0.4.1 torchvision cuda90 -c pytorch
conda install torchvision=0.2.1 cuda90

# Install the rest of the dependencies with pip.
pip install numpy six future forbiddenfruit
```

Every time you run code in the repo, you should:
```shell
conda activate diffai
```

## Certification

First, download the `convSmall` trained model from DiffAI. Then you can run:

```shell
python . -d Point --epochs 1 --dont-write --test-freq 1 -t Box(0.01) --test out/convSmall.pynet -D=CIFAR10
```

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