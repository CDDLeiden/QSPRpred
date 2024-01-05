## Table of Contents

This is a list of tutorials for QSPRpred.

The tutorial data is available
through [OneDrive](https://1drv.ms/u/s!AtzWqu0inkjX3QRxXOkTFNv7IV7u?e=PPj0O2) (just
unzip and place the two datasets `A2A_LIGANDS.tsv` and `AR_LIGANDS.tsv` in
the `tutorial_data` folder) or recreate the dataset yourself by
running `tutorial_data/create_tutorial_data.py`.

A Quick Start tutorial is designed to get you up and running with QSPRpred as quickly as
possible.
The rest of the tutorials is divided into two categories: basic and advanced.
The Basics tutorials cover all basic functionality of QSPRpred that you will need in
most projects. The Advanced tutorials cover more advanced topics and are designed for
users who are already familiar with the basics of QSPRpred.
For detailed description of all QSPRpred classes and functions, see
the [documentation](https://cddleiden.github.io/QSPRpred/docs/).

- **[Quick Start](quick_start.ipynb)**: A quick start guide to using QSPRpred.
- **Basics**
    - Data
        - [Data Collection with Papyrus](basics/data/data_collection_with_papyrus.ipynb):
          How to collect data with Papyrus.
        - [Data Preparation](basics/data/data_preparation.ipynb): How to prepare data
          for QSPRpred.
        - [Data Representation](basics/data/data_representation.ipynb): How data is
          represented in QSPRpred (MolTable, QSPRDataset, etc.).
        - [Data Splitting](basics/data/data_splitting.ipynb): How to split data into
          training, validation, and test sets.
        - [Descriptors](basics/data/descriptors.ipynb): How to calculate descriptors for
          molecules.
    - Modelling
        - [Classification](basics/modelling/classification.ipynb): How to train a
          classification model.
        - [Logging](basics/modelling/logging.ipynb): How to set-up logging.
        - [Model Assessment](basics/modelling/model_assessment.ipynb): How to assess the
          performance of a model.
    - Benchmarking
        - [Benchmarking](basics/benchmarking/benchmarking.ipynb): How to benchmark
          QSPRpred.
- **Advanced**
    - Data
        - [Custom descriptors](advanced/data/custom_descriptors.ipynb): How to use
          custom descriptors.
        - [Custom data splitting](advanced/data/custom_splitting.ipynb): How to use
          custom data splitting.
    - Modelling
        - [Custom models](advanced/modelling/custom_models.ipynb): How to use custom
          models.
        - [Deep learning models](advanced/modelling/deep_learning_models.ipynb): How to
          use deep learning models.
        - [Hyperparameter optimization](advanced/modelling/hyperparameter_optimization.ipynb):
          How to optimize model hyperparameters.
        - [Monitoring](advanced/modelling/monitoring.ipynb): How to monitor model
          training.
        - [Multi-task learning](advanced/modelling/multi_task_modelling.ipynb): How to
          train a multi-task model.
        - [PCM modelling](advanced/modelling/PCM_modelling.ipynb): How to prepare data
          for and train a proteochemometric model.