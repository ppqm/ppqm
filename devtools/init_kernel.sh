#!/bin/bash

ENVNAME=lab

source activate $ENVNAME

conda install ipykernel

python -m ipykernel install --user --name ppqm --display-name "PPQM (Python 3.9)"
