[![GitHub Marketplace](https://img.shields.io/badge/Marketplace-autopep8-blue.svg?colorA=24292e&colorB=0366d6&style=flat&longCache=true&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAM6wAADOsB5dZE0gAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAERSURBVCiRhZG/SsMxFEZPfsVJ61jbxaF0cRQRcRJ9hlYn30IHN/+9iquDCOIsblIrOjqKgy5aKoJQj4O3EEtbPwhJbr6Te28CmdSKeqzeqr0YbfVIrTBKakvtOl5dtTkK+v4HfA9PEyBFCY9AGVgCBLaBp1jPAyfAJ/AAdIEG0dNAiyP7+K1qIfMdonZic6+WJoBJvQlvuwDqcXadUuqPA1NKAlexbRTAIMvMOCjTbMwl1LtI/6KWJ5Q6rT6Ht1MA58AX8Apcqqt5r2qhrgAXQC3CZ6i1+KMd9TRu3MvA3aH/fFPnBodb6oe6HM8+lYHrGdRXW8M9bMZtPXUji69lmf5Cmamq7quNLFZXD9Rq7v0Bpc1o/tp0fisAAAAASUVORK5CYII=)](https://github.com/marketplace/actions/autopep8)

QSPRpred
====================

This repository can be used for building **Quantitative Structure Property Relationship (QSPR)** models.
It is based on the QSAR models in **Drug Explorer (DrugEx)**, a _de novo_ drug design tool based on deep learning,
originally developed by [Xuhan Liu](https://github.com/XuhanLiu/DrugEx/) & Gerard J.P. van Westen. [1,2,3] and further
developed by [Gerard van Westen's Computational Drug Discovery group](https://github.com/CDDLeiden/DrugEx).
Models developed with QSPRpred are compatible with the CDD DrugEx repo. 

Quick Start
===========

### Installation

QSPRpred can be installed with pip like so (with python >= 3.9.0):

```bash
pip install git+https://github.com/CDDLeiden/QSPRPred.git@main
```

### Use
After installation, you will have access to various command line features, but you can also use the Python API directly. Documentation for the current version of both is available [here](https://cddleiden.github.io/QSPRpred/docs/). For a quick start, you can also check out the  [Jupyter notebook tutorials](./tutorial), which documents the use of the Python API to build different types of models. [This tutorial](./tutorial/tutorial_training.ipynb) shows how a QSAR model can be trained. [This tutorial](./tutorial/tutorial_usage.ipynb) shows how to use a QSAR model to predict the bioactivity of a set of molecules. The tutorials as well as the documentation are still work in progress, and we will be happy for any contributions where it is still lacking.

To use the commandline to train the same QSAR model as in the tutorial use:
```bash
python -m qsprpred.QSPR_cli -i parkinsons_pivot.tsv -pr GABAAalpha -m XGB -r true -sf 0.15 -fe Morgan -s -o bayes -nt 10 -me
```

Current Development Team
========================
- S. Luukkonen
- M. Sicho
- H. van den Maagdenberg
- L. Schoenmaker

Acknowledgements
================

We would like to thank the following people for significant contributions:

- Xuhan Liu
  - author of the original idea to develop the DrugEx models and code, we are happy for his continuous support of the project

References
==========

1. [Liu X, Ye K, van Vlijmen HWT, IJzerman AP, van Westen GJP. DrugEx v3: Scaffold-Constrained Drug Design with Graph Transformer-based Reinforcement Learning. Preprint](https://chemrxiv.org/engage/chemrxiv/article-details/61aa8b58bc299c0b30887f80)

2. [Liu X, Ye K, van Vlijmen HWT, Emmerich MTM, IJzerman AP, van Westen GJP. DrugEx v2: De Novo Design of Drug Molecule by Pareto-based Multi-Objective Reinforcement Learning in Polypharmacology. Journal of cheminformatics 2021:13(1):85.](https://doi.org/10.1186/s13321-021-00561-9) 

3. [Liu X, Ye K, van Vlijmen HWT, IJzerman AP, van Westen GJP. An exploration strategy improves the diversity of de novo ligands using deep reinforcement learning: a case for the adenosine A2A receptor. Journal of cheminformatics. 2019;11(1):35.](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0355-6)


