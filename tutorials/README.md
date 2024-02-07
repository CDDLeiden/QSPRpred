## Table of Contents

This
tutorial tries to cover the most important topics on the various features of QSPRpred,
but it is not exhaustive. For more detailed information on the features of the package,
please refer to the [documentation](https://cddleiden.github.io/QSPRpred/docs/).The
tutorial data is available
through [OneDrive](https://1drv.ms/u/s!AtzWqu0inkjX3QRxXOkTFNv7IV7u?e=PPj0O2) (just
unzip and place the two datasets `A2A_LIGANDS.tsv` and `AR_LIGANDS.tsv` in
the `tutorial_data` folder) or recreate the dataset yourself by
running `tutorial_data/create_tutorial_data.py` after you have installed QSPRpred.

The [Quick Start](quick_start.ipynb) tutorial is designed to get you up and running with
QSPRpred as quickly as
possible while the rest dedicates more time to explain each feature in more detail.
The [Basics](./basics) cover the most commonly used functionality of QSPRpred.
The [Advanced](./advanced) tutorials cover more advanced topics and are designed for
users who are already familiar with QSPRpred more in depth or are looking for more niche
features. For detailed description of all QSPRpred classes and functions, as well as
examples of
how to use the command line interface, see
the [documentation pages](https://cddleiden.github.io/QSPRpred/docs/).

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
        - [Searching, Filtering and Plotting](basics/data/searching_filtering_plotting.ipynb):
          How to search and filter data.
        - [Applicability Domain](basics/data/applicability_domain.ipynb): How to
          calculate the applicability domain of a model.
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
        - [Parallelization](advanced/data/parallelization.ipynb): How to parallelize
          data functions across data sets.
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
        - [Chemprop models](advanced/modelling/chemprop_models.ipynb): How to use
          Chemprop models in QSPRpred.