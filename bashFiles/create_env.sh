#!/bin/bash

#echo "Deleting ENN-ENV"
#eval "bash"
#eval "yes | conda deactivate"
#eval "yes | conda env remove -n enn-env"

echo "Creating new ENN-ENV"
eval "yes | conda create -n enn-env python=3.11"
eval "conda activate enn-env"

echo "Installing packages"
eval "yes | conda install pip"
eval "yes | conda install pandas"
eval "yes | conda install matplotlib"
eval "yes | conda install jupyter"
eval "yes | conda install conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"

echo "done"