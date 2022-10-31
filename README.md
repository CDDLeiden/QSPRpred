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

QSPRpred can be installed with pip like so:

```bash
pip install git+https://github.com/CDDLeiden/QSPRpred.git@master
```

### Use
After installation, you will have access to various command line features, but you can also use the Python API directly. Documentation for the current version of both is available [here](https://cddleiden.github.io/QSPRpred/docs/). For a quick start, you can also check out our [Jupyter notebook tutorial](./tutorial), which documents the use of the Python API to build different types of models. The tutorial as well as the documentation are still work in progress, and we will be happy for any contributions where it is still lacking.

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