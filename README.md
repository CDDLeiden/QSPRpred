[![GitHub Marketplace](https://img.shields.io/badge/Marketplace-autopep8-blue.svg?colorA=24292e&colorB=0366d6&style=flat&longCache=true&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAM6wAADOsB5dZE0gAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAERSURBVCiRhZG/SsMxFEZPfsVJ61jbxaF0cRQRcRJ9hlYn30IHN/+9iquDCOIsblIrOjqKgy5aKoJQj4O3EEtbPwhJbr6Te28CmdSKeqzeqr0YbfVIrTBKakvtOl5dtTkK+v4HfA9PEyBFCY9AGVgCBLaBp1jPAyfAJ/AAdIEG0dNAiyP7+K1qIfMdonZic6+WJoBJvQlvuwDqcXadUuqPA1NKAlexbRTAIMvMOCjTbMwl1LtI/6KWJ5Q6rT6Ht1MA58AX8Apcqqt5r2qhrgAXQC3CZ6i1+KMd9TRu3MvA3aH/fFPnBodb6oe6HM8+lYHrGdRXW8M9bMZtPXUji69lmf5Cmamq7quNLFZXD9Rq7v0Bpc1o/tp0fisAAAAASUVORK5CYII=)](https://github.com/marketplace/actions/autopep8)

QSPRpred
====================

<img src='figures/QSPRpred_logo.jpg' width=10% align=right>
<p align=left width=70%>
QSPRpred is open-source software libary for building **Quantitative Structure Property Relationship (QSPR)** model developed by Gerard van Westen's Computational Drug Discovery group. Models developed with QSPRpred are compatible with the group's *de novo* drug design package <a href="https://github.com/CDDLeiden/DrugEx/">DrugEx</a>.

<!-- This repository can be used for building **Quantitative Structure Property Relationship (QSPR)** models.
It is based on the QSAR models in **Drug Explorer (DrugEx)**, a _de novo_ drug design tool based on deep learning,
originally developed by [Xuhan Liu](https://github.com/XuhanLiu/DrugEx/) & Gerard J.P. van Westen. [1,2,3] and further
developed by [Gerard van Westen's Computational Drug Discovery group](https://github.com/CDDLeiden/DrugEx).
Models developed with QSPRpred are compatible with the CDD DrugEx repo.  -->

Quick Start
===========

## Installation

QSPRpred can be installed with pip like so (with python >= 3.8.0):

```bash
pip install git+https://github.com/CDDLeiden/QSPRPred.git@main
```

If you plan to optionally use QSPRPred to calculate PCM descriptors, make sure to also install Clustal Omega. You can get it via `conda`:

```bash

conda install -c bioconda clustalo
```

## Use
After installation, you will have access to various command line features, but you can also use the Python API directly (see [Documentation](https://cddleiden.github.io/QSPRPred/docs/)). For a quick start, you can also check out the  [Jupyter notebook tutorials](./tutorial), which documents the use of the Python API to build different types of models. [This tutorial](./tutorial/tutorial_training.ipynb) shows how a QSAR model can be trained. [This tutorial](./tutorial/tutorial_usage.ipynb) shows how to use a QSAR model to predict the bioactivity of a set of molecules. The tutorials as well as the documentation are still work in progress, and we will be happy for any contributions where it is still lacking.

To use the commandline to train the same QSAR model as in the tutorial use (run from tutorial folder):
```bash
python -m qsprpred.data_CLI -i parkinsons_pivot.tsv -pr GABAAalpha -r true -sf 0.15 -fe Morgan
python -m qsprpred.model_CLI -pr GABAAalpha -r true -m PLS -s -o bayes -nt 10 -me
```

Workflow
========
![image](figures/QSPRpred_workflow.png)

Current Development Team
========================
- [H. van den Maagdenberg](https://github.com/HellevdM)
- [M. Sicho](https://github.com/martin-sicho)
- [L. Schoenmaker](https://github.com/LindeSchoenmaker)
- [O. Béquignon](https://github.com/OlivierBeq)
- [S. Luukkonen](https://github.com/sohviluukkonen)
- [M. Gorosiola González](https://github.com/gorostiolam)
- [D. Araripe](https://github.com/David-Araripe)
