..  _usage:

Usage
=====

You can use the command-line interface to preprocess data and build models.
The description of the functionality can be displayed with the :code:`--help` argument.
The help message for the :code:`QSPRpred.QSPR_cli` script can be shown as follows:

..  code-block::

    python -m qsprpred.QSPR_cli --help

A simple command-line workflow to prepare your dataset and train QSPR model is given below (see :ref:`cli-example`).

If you want more control over the inputs and outputs or want to customize QSPRpred a bit more,
you can also use the Python API directly (see :ref:`source code <https://github.com/CDDLeiden/QSPRpred/tree/master/tutorial>`_).
Here you can find a tutorial with a Jupyter notebook illustrating some common use cases in the project source code.

..  _cli-example:

CLI Example
===========

.. _basics:

Basics
------

In this example, we will use the command line utilities of QSPRpred to train a QSAR model for the Adenosine 1A and Receptor and the 
Adenosine 2A Receptor on data from the CHEMBL 27 database.
We use the same data from `the DrugEx tutorial <https://drive.google.com/file/d/1lYOmQBnAawnDR2Kwcy8yVARQTVzYDelw/view>`_ here, but you first need to make sure
that the data is in the right format. See :ref:`source code <https://github.com/CDDLeiden/QSPRpred/tree/master/tutorial>` for an example on how to do this.

Preparing Data
^^^^^^^^^^^^^^^

QSPRpred assumes that all input data are saved in the data folder of the directory it is executed from.
Therefore, we place our dataset in a folder 'data'. In the CLI we also need to indicate which property/ies we are interested in predicting (here CHEMBL226, CHEMBL251, the columns with 
bioactivity values for Adenosine receptor A1 and Adenosine receptor A2a, respectively), this should be equal to the column names containing the values to be predicted in. 
These column names should also not contain any spaces.
For regression models these columns should contain numerical datapoints. For categorical models either categorical data or numerical data can be used (the latter will be categorized based on the activity threshold).
Furthermore, we should indicate how we wish to split the data to create a train and test set.
Here we will use a random split with a test fraction of 15%. We need to calculate feature from the SMILES sequence, here we use morgan fingerprints and rdkit descriptors.

Model Training
^^^^^^^^^^^^^^

Finally, we need to indicate what models we want to train and which steps to take in the training.
In this example, we will build classification random forest models and naive bayes models.
We will also do model evalutation through cross-validation and finally train the model on all data and save.

..  code-block::

    # input is in ./data/LIGAND_RAW_small.tsv
        qsprpred QSPR_cli -i LIGAND_RAW_small.tsv -pr CHEMBL226 -pr CHEMBL251 -sp random -sf 0.15 -fe Morgan RDkit -m RF NB -me -s


CLI Options
-----------

There are many options to choose from when training your QSPR model from the CLI.
Here you can find a short overview.

Other
^^^^^


