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
you can also use the Python API directly (see :ref:`source code <https://github.com/CDDLeiden/QSPRpred/tree/master/tutorial>`).
Here you can find a tutorial with a Jupyter notebook illustrating some common use cases in the project source code.

..  _cli-example:

CLI Example
===========

.. _basics:

Basics
------

In this example, we will use the command line utilities of QSPRpred to train a QSAR model for the Adenosine 1A and Receptor and the 
Adenosine 2A Receptor on data from the CHEMBL 27 database.
We use the same data from `the DrugEx tutorial <https://drive.google.com/file/d/1lYOmQBnAawnDR2Kwcy8yVARQTVzYDelw/view>` here, but you first need to make sure
that the data is in the right format. See :ref:`source code <https://github.com/CDDLeiden/QSPRpred/tree/master/tutorial>` for an example on how to do this.

Preparing Data
^^^^^^^^^^^^^^^

QSPRpred assumes that all input data are saved in the data folder of the directory it is executed from.
Therefore, we place our dataset in a folder 'data'. In the CLI we also need to indicate which property/ies we are interested in predicting (here CHEMBL226, CHEMBL251, the columns with 
bioactivity values for Adenosine receptor A1 and Adenosine receptor A2a, respectively), this should be equal to the column names containing the values to be predicted in. 
These column names should also not contain any spaces.
For regression models these columns should contain numerical datapoints. For categorical models either categorical data or numerical data can be used (the latter will be categorized based on the activity threshold).
Furthermore, we should indicate how we wish to split the data to create a train and test set.
Here we will use a random split with a test fraction of 15%. We need to calculate feature from the SMILES sequence, here we use morgan fingerprints.

Model Training
^^^^^^^^^^^^^^

Finally, we need to indicate what models we want to train and which steps to take in the training.
In this example, we will build regression random forest models.
We will also do model evaluation through cross-validation and finally train the model on all data and save.

..  code-block::

    # input is in ./data/LIGAND_RAW_small.tsv
        qsprpred QSPR_cli -i LIGAND_RAW_small.tsv -pr CHEMBL226 -pr CHEMBL251 -r REG -sp random -sf 0.15 -fe Morgan -m RF -me -s


CLI Options
-----------

There are many options to choose from when training your QSPR model from the CLI.
Here you can find a short overview.


Base settings arguments
^^^^^^^^^^^^^^^^^^^^^^^
Apart from the base directory and the input file, there are a few other base options that
can be set. Including the -d, will also print debug options to the log file. The random 
seed can also be set manually (although identical results are not guaranteed with keeping
the same random seed). Furthermore, the number of cpu's used for model training and the
gpu number for training pytorch models can be set. Finally, the name of the smilescolumn
in your dataset can be indicated.

..  code-block::

    # Setting debug flag, smiles column, random seed, number of cpu's and a specific gpu (for now multiple gpu's not possible)
        qsprpred QSPR_cli -i LIGAND_RAW_small.tsv -d -ran 42 -ncpu 5 -gpus [3] -sm Smiles -pr CHEMBL226 -pr CHEMBL251 -sp random -sf 0.15 -fe Morgan -m RF -me -s


Data arguments
^^^^^^^^^^^^^^
Log-transform data
""""""""""""""""""
To log transform data specific properties, indicate this in the CLI as follows:

..  code-block::

    # Log transform data for CHEMBL226
    qsprpred QSPR_cli -i LIGAND_RAW_small.tsv -pr CHEMBL226 -pr CHEMBL251 -lt '{"CHEMBL226":true,"CHEMBL251":false}' -r REG -sp random -sf 0.15 -fe Morgan -m RF -me -s

Train test split
""""""""""""""""
In the base example we use a random split to create the train and test set. There are two
more options, namely a scaffold split, where the data is split into a test and train set
randomly but keeping molecules with the same Murcko scaffold in the same set.

..  code-block::

    # input is in ./data/LIGAND_RAW_small.tsv
        qsprpred QSPR_cli -i LIGAND_RAW_small.tsv -sm Smiles -pr CHEMBL226 -pr CHEMBL251 -sp scaffold -sf 0.15 -fe Morgan -m RF -me -s

The third option is a temporal split, where a column needs to be indicated which holds
information on the time each sample was observed and split based on threshold in a column.
In this example, all samples after 2015 (in column 'year') make up the test set.

..  code-block::

    # input is in ./data/LIGAND_RAW_small.tsv
        qsprpred QSPR_cli -i LIGAND_RAW_small.tsv -sm Smiles -pr CHEMBL226 -pr CHEMBL251 -sp time -st 2015 -stc year 0.15 -fe Morgan -m RF -me -s

Feature calculation
"""""""""""""""""""
There are four different descriptor sets that can be calculated at the moment,
namely Morgan fingerprints, rdkit descriptors, Mordred descriptors and the
physicochemical properties used in the QSAR models in the DrugEx papers. The can also
be combined. For more control over the descriptorcalculator settings use the python API.

..  code-block::

    # input is in ./data/LIGAND_RAW_small.tsv
        qsprpred QSPR_cli -i LIGAND_RAW_small.tsv -pr CHEMBL226 -pr CHEMBL251 -r REG -sp random -sf 0.15 -fe Morgan RDkit Mordred DrugEx -m RF -me -s

Feature filtering
"""""""""""""""""
The calculated features can also be filtered. Three different filters are implemented in
QSPRpred, namely a high correlation filter, a low variance filter and the boruta filter.
The high correlation filter and low variance filter need to be set with a threshold
for filtering.

..  code-block::

    # input is in ./data/LIGAND_RAW_small.tsv
        qsprpred QSPR_cli -i LIGAND_RAW_small.tsv -pr CHEMBL226 -pr CHEMBL251 -r REG -sp random -sf 0.15 -fe Morgan -lv 0.1 -hc 0.9 -bf -m RF -me -s

Papyrus Low quality filter
""""""""""""""""""
Specifically for use with a dataset from the `Papyrus dataset <https://chemrxiv.org/engage/chemrxiv/article-details/617aa2467a002162403d71f0>`,
an option is included for filtering low quality data from the dataset. 
(All data is removed with value 'Low' in column 'Quality')


Model arguments
^^^^^^^^^^^^^^^

Classification models
"""""""""""""""""""""
The model training can be customized with several CLI arguments.
Firstly, you can set whether to use regression, classification or both.
The default setting is to run both, but you can run either by setting the
regression argument to true/REG for regression or false/CLS for classification.
When using classification, the threshold(s) for each property need to be included.
This is set using a dictionary. In case of multi-class classification the bounderies of
the bins need to be given. For binary only give 1 threshold per property.

..  code-block::

    # input is in ./data/LIGAND_RAW_small.tsv
        qsprpred QSPR_cli -i LIGAND_RAW_small.tsv -pr CHEMBL226 -pr CHEMBL251 -r CLS -th '{"CHEMBL226":[6.5],"CHEMBL251":[0, 3, 6, 10]}' -sp random -sf 0.15 -fe Morgan -m RF -me -s


model types
"""""""""""
You also need to indicate which models you want to run, out of the following model types:
'RF' (Random Forest), 'XGB' (XGboost), 'SVM' (Support Vector Machine), 'PLS' (partial least squares regression),
'KNN' (k-nearest neighbours), NB' (Naive Bayes) and/or 'DNN' (pytorch fully connected neural net).
The default is to run all the different model types.

..  code-block::

    # input is in ./data/LIGAND_RAW_small.tsv
        qsprpred QSPR_cli -i LIGAND_RAW_small.tsv -pr CHEMBL226 -pr CHEMBL251 -r REG -sp random -sf 0.15 -fe Morgan -m RF SVM NB -me -s

Defining model parameters
"""""""""""""""""""""""""
Specific model parameters can be set with the parameters argument by giving a json file.
Specifically for the training of the DNN model, you can set the tolerance and the patience from the CLI.
Tolerance gives the mimimum decrease in loss needed to count as an improvement and 
patience is the number of training epochs without improvement in loss to stop the training.

..  code-block::

    # input is in ./data/LIGAND_RAW_small.tsv
        qsprpred QSPR_cli -i LIGAND_RAW_small.tsv -pr CHEMBL226 -pr CHEMBL251 -r REG  -p myparams -sp random -sf 0.15 -fe Morgan -m RF -me -s

..  code-block::

    # input is in ./data/LIGAND_RAW_small.tsv
        qsprpred QSPR_cli -i LIGAND_RAW_small.tsv -pr CHEMBL226 -pr CHEMBL251 -r REG -sp random -sf 0.15 -fe Morgan -m DNN -tol 0.02 -pat 100 -me -s

Hyperparameter optimization
"""""""""""""""""""""""""""
In addition to setting model parameters manually, a hyperparameter search can be performed.
In QSPRpred, two methods of hyperparameter optimization are implemented: grid search and 
bayesian optimization. For baysian optimization also give the number of trials.
The search space needs to be set using a json file, if this is not given then the default
search space defined in qsprpred/models/search_space.json is used.
A simple search space file for a RF and KNN model should look as given below.
Note the indication of the model type as first list item and type of optimization algorithm
as third list item. The search space file should always include all models to be trained.

..  code-block::

    [["RF", {"max_depth": [null, 20, 50, 100],
            "max_features": ["sqrt", "log2"],
            "min_samples_leaf": [1, 3, 5]}, "grid"],
    ["RF", {"n_estimators": ["int", 10, 2000],
            "max_depth": ["int", 1, 100],
            "min_samples_leaf": ["int", 1, 25]}, "bayes"],
    ["KNN", {"n_neighbors" : [1, 5, 15, 25, 30],
            "weights"      : ["uniform", "distance"]}, "grid"],
    ["KNN", {"n_neighbors": ["int", 1, 100],
            "weights": ["categorical", ["uniform", "distance"]],
            "metric": ["categorical", ["euclidean","manhattan",
                        "chebyshev","minkowski"]]}, "bayes"]]

..  code-block::

    # input is in ./data/LIGAND_RAW_small.tsv
        qsprpred QSPR_cli -i LIGAND_RAW_small.tsv -pr CHEMBL226 -pr CHEMBL251 -r REG -sp random -sf 0.15 -fe Morgan -m RF -o bayes -nt 50 -ss mysearchspace.json -me -s

