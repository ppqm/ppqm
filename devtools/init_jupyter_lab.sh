#!/bin/bash

ENVNAME=lab

pythonversion=3.8
nglviewversion=2.7.7
labversion=2.1.5
ipywidgetsversion=7.5.1

conda create -n $ENVNAME python=$pythonversion -y
source activate $ENVNAME

conda install ipywidgets==$ipywidgetsversion -c conda-forge -y
conda install nglview==$nglviewversion -c conda-forge -y
conda install jupyterlab=$labversion  -y -c conda-forge
jupyter-labextension install @jupyter-widgets/jupyterlab-manager
jupyter-labextension install nglview-js-widgets@$nglviewversion
