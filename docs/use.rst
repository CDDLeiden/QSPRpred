..  _usage:

Usage
=====

You can use the command-line interface to preprocess data and build models.
The description of the functionality can be displayed with the :code:`--help` argument,
e.g. the help message for the :code:`QSPRpred.data_CLI` script can be shown as follows:

..  code-block::

    python -m qsprpred.data_CLI --help

A simple command-line workflow to prepare your dataset and train QSPR models is given below (see :ref:`cli-example`).

If you want more control over the inputs and outputs or want to customize QSPRpred a bit more,
you can also use the Python API directly (see :ref:`source code <https://github.com/CDDLeiden/QSPRpred/tree/master/tutorial>`).
Here you can find a tutorial with a Jupyter notebook illustrating some common use cases in the project source code.

..  _cli-example:

CLI Example
===========

In this example, we will use the command line utilities of QSPRpred to train a QSAR model for the Adenosine A1 receptor and the 
Adenosine A2 Receptor on data from the CHEMBL 27 database.
We use the same data from `the DrugEx tutorial <https://drive.google.com/file/d/1lYOmQBnAawnDR2Kwcy8yVARQTVzYDelw/view>` here, but you first need to make sure
that the data is in the right format. See :ref:`source code <https://github.com/CDDLeiden/QSPRpred/tree/master/tutorial>` for an example on how to do this.
QSPRpred assumes that all input data are saved in the data folder of the directory it is executed from.
Therefore, our example dataset is located in a subfolder 'data' of the tutorial directory.

Preparing Data
--------------

Basics
^^^^^^
We will now use the QSPRpred :code:`QSPRpred.data_CLI` script for data preparation.
In the CLI we need to indicate which property/ies we are interested in predicting (here CHEMBL318, CHEMBL256, the columns with 
bioactivity values for Adenosine receptor A1 and Adenosine receptor A3, respectively), this should be equal to the column names containing the values to be predicted. 
These column names should also not contain any spaces.
For regression models these columns should contain numerical datapoints. For categorical models either categorical data or numerical data can be used (the latter will be categorized based on the activity threshold).
Furthermore, we should indicate how we wish to split the data to create a train and test set.
Here we will use a random split with a test fraction of 15%. We need to calculate feature from the SMILES sequence, here we use morgan fingerprints.

..  code-block::

    # input is in ./data/LIGAND_RAW_small_pivot.tsv
        python -m qsprpred.data_CLI -i LIGAND_RAW_small_pivot.tsv -sm Smiles -pr CHEMBL318 -pr CHEMBL256 -r REG -sp random -sf 0.15 -fe Morgan

Running this command will create a subfolder 'qspr'. The qspr/data folder should contains 8 files for
each property (16 in total), prefixed by the property identifiers (i.e. CHEMBL318/CHEMBLE256), the model task (REG/CLS).
As well as an log file and a run settings file.

+--------------------------------------------------+-------------------------------------------------------+
| File                                             | Function                                              |
+==================================================+=======================================================+
|| {prefixes}_QSPRdata_feature_calculators.json    || instantiate feature calculator                       |
|| {prefixes}_QSPRdata_feature_standardizer_0.json || instantiate feature standardizer                     |
|| {prefixes}_QSPRdata_meta.json                   || Meta data, also used to instantiate QSAR data object |
|| {prefixes}_QSPRdata_X_ind.pkl                   || Independent test set input                           |
|| {prefixes}_QSPRdata_y_ind.pkl                   || Independent test set output                          |
|| {prefixes}_QSPRdata_X.pkl                       || Training set input                                   |
|| {prefixes}_QSPRdata_y.pkl                       || Training set output                                  |
|| {prefixes}_QSPRdata_df.pkl                      || Input dataframe                                      |
|| QSPRdata.json                                   || Command Line interface settings                      |
|| QSPRdata.log                                    || Log file                                             |
+--------------------------------------------------+-------------------------------------------------------+


More
^^^^
Run settings arguments
^^^^^^^^^^^^^^^^^^^^^^^
Apart from the base directory and the input file, there are a few other base options that
can be set. Including `-d`, will print debug information to the log file. The random 
seed can also be set manually (although identical results are not guaranteed while keeping
the same random seed). Furthermore, the number of cpu's used for model training. Finally, the name of the smilescolumn
in your dataset can be indicated.

..  code-block::

    # Setting debug flag, smiles column, random seed, number of cpu's
        python -m qsprpred.data_CLI -i LIGAND_RAW_small_pivot.tsv -d -ran 42 -ncpu 5 -gpus [3] -sm Smiles -pr CHEMBL318 -pr CHEMBL256 -r REG -sp random -sf 0.15 -fe Morgan


Log-transform data
""""""""""""""""""
To log transform data specific properties, indicate this in the CLI as follows:

..  code-block::

    # Log transform data for CHEMBL318
    python -m qsprpred.data_CLI -i LIGAND_RAW_small_pivot.tsv -sm Smiles -pr CHEMBL318 -pr CHEMBL256 -lt '{"CHEMBL318":true,"CHEMBL256":false}' -r REG -sp random -sf 0.15 -fe Morgan

Train test split
""""""""""""""""
In the base example we use a random split to create the train and test set. There are two
more options, namely a scaffold split, where the data is split into a test and train set
randomly but keeping molecules with the same Murcko scaffold in the same set.

..  code-block::

    # Scaffold split
        python -m qsprpred.data_CLI -i LIGAND_RAW_small_pivot.tsv -sm Smiles -pr CHEMBL318 -pr CHEMBL256 -sp scaffold -sf 0.15 -fe Morgan

The third option is a temporal split, where a column needs to be indicated which holds
information on the time each sample was observed and split based on threshold in a column.
In this example, all samples after 2015 (in column 'year') make up the test set.

..  code-block::

    # Time split
        python -m qsprpred.data_CLI -i LIGAND_RAW_small_pivot.tsv -sm Smiles -pr CHEMBL318 -pr CHEMBL256 -sp time -st 2015 -stc year 0.15 -fe Morgan


Data for classification models
""""""""""""""""""""""""""""""
You can set whether to prepare data for regression, classification or both.
The default setting is to run both, but you can run either by setting the
regression argument to true/REG for regression or false/CLS for classification.
When using classification, the threshold(s) for each property need to be included.
This is set using a dictionary. In case of multi-class classification the bounderies of
the bins need to be given. For binary only give 1 threshold per property.

..  code-block::

    # Classification and regression
        python -m qsprpred.data_CLI -i LIGAND_RAW_small_pivot.tsv -sm Smiles -pr CHEMBL318 -pr CHEMBL256 -r CLS -sp random -sf 0.15 -fe Morgan -th '{"CHEMBL318":[6.5],"CHEMBL256":[0, 3, 6, 10]}'

Feature calculation
"""""""""""""""""""
There are four different descriptor sets that can be calculated at the moment,
namely Morgan fingerprints, rdkit descriptors, Mordred descriptors and the
physicochemical properties used in the QSAR models in the DrugEx papers. The can also
be combined. For more control over the descriptorcalculator settings use the python API.

..  code-block::

    # With Morgan, RDkit, Mordred and DrugEx descriptors
        python -m qsprpred.data_CLI -i LIGAND_RAW_small_pivot.tsv -sm Smiles -pr CHEMBL318 -pr CHEMBL256 -r REG -sp random -sf 0.15 -fe Morgan RDkit Mordred DrugEx

Feature filtering
"""""""""""""""""
The calculated features can also be filtered. Three different filters are implemented in
QSPRpred, namely a high correlation filter, a low variance filter and the boruta filter.
The high correlation filter and low variance filter need to be set with a threshold
for filtering.

..  code-block::

    # input is in ./data/LIGAND_RAW_small.tsv
       python -m qsprpred.data_CLI -i LIGAND_RAW_small_pivot.tsv -sm Smiles -pr CHEMBL318 -pr CHEMBL256 -r REG -sp random -sf 0.15 -fe Morgan -lv 0.1 -hc 0.9 -bf

Papyrus Low quality filter
""""""""""""""""""""""""""
Specifically for use with a dataset from the `Papyrus dataset <https://chemrxiv.org/engage/chemrxiv/article-details/617aa2467a002162403d71f0>`,
an option is included for filtering low quality data from the dataset (All data is removed with value 'Low' in column 'Quality').
To apply this filter include `-lq` or `--low_quality` in your command.

Model Training
--------------

Basics
^^^^^^

Finally, we need to indicate what models we want to train and which steps to take in the training.
In this example, we will build regression random forest models.
We will also evaluate the model through cross-validation and train the model on all data to save for further use.

..  code-block::

    # input is in ./data/LIGAND_RAW_small_pivot.tsv
        python -m qsprpred.model_CLI -pr CHEMBL318 -pr CHEMBL256 -r REG -m RF -me -s

More
^^^^
The model training can be further customized with several CLI arguments.
Here you can find a short overview.

run settings arguments
^^^^^^^^^^^^^^^^^^^^^^^
As with the data preparation including `-d`, will print debug information to the log file. The random 
seed can also be set manually (although identical results are not guaranteed while keeping
the same random seed). Furthermore, the number of cpu's used for model training and the
gpu number for training pytorch models can be set.

..  code-block::

    # Setting debug flag, random seed, number of cpu's and a specific gpu (for now multiple gpu's not possible)
        python -m qsprpred.model_CLI -d -ran 42 -ncpu 5 -gpus [3] -pr CHEMBL318 -pr CHEMBL256 -r REG -m RF -me -s

Classification models
"""""""""""""""""""""

Firstly, you can set whether to use regression, classification or both.
The default setting is to run both, but you can run either by setting the
regression argument to true/REG for regression or false/CLS for classification.
Make sure you have prepared datasets for the corresponding tasks.

..  code-block::

    # Training a classification model
        python -m qsprpred.model_CLI -pr CHEMBL318 -pr CHEMBL256 -r CLS -m RF -me -s

model types
"""""""""""
You also need to indicate which models you want to run, out of the following model types:
'RF' (Random Forest), 'XGB' (XGboost), 'SVM' (Support Vector Machine), 'PLS' (partial least squares regression),
'KNN' (k-nearest neighbours), NB' (Naive Bayes) and/or 'DNN' (pytorch fully connected neural net).
The default is to run all the different model types.

..  code-block::

    # Training a RF, SVM and PLS model
        python -m qsprpred.model_CLI -pr CHEMBL318 -pr CHEMBL256 -r REG -me -s -m RF SVM PLS

Defining model parameters
"""""""""""""""""""""""""
Specific model parameters can be set with the parameters argument by passing a json file.

./myparams.json
..  code-block::

    [["RF", {"max_depth": [null, 20, 50, 100],
            "max_features": ["sqrt", "log2"],
            "min_samples_leaf": [1, 3, 5]}],
    ["KNN", {"n_neighbors" : [1, 5, 15, 25, 30],
            "weights"      : ["uniform", "distance"]}]]

..  code-block::

    # Setting some parameter values for a Random Forest and k-nearest neighbours model
        python -m qsprpred.model_CLI -pr CHEMBL318 -pr CHEMBL256 -r REG -m RF KNN -me -s -p myparams

Specifically for the training of the DNN model, you can set the tolerance and the patience from the CLI.
Tolerance gives the mimimum decrease in loss needed to count as an improvement and 
patience is the number of training epochs without improvement in loss to stop the training.

..  code-block::

    # Setting the tolerance and patience for training a DNN model
        python -m qsprpred.model_CLI -pr CHEMBL318 -pr CHEMBL256 -r REG -me -s -m DNN -tol 0.02 -pat 100

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

./mysearchspace.json
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

    # Bayesian optimization
        python -m qsprpred.model_CLI -pr CHEMBL318 -pr CHEMBL256 -r REG -m RF -me -s -o bayes -nt 50 -ss mysearchspace -me -s


Prediction
-----------
Furthermore, trained QSPRpred models can be used to predict values from SMILES from the command line interface :code:`predict_CLI.py`.

Basics
^^^^^^
Here we will predict activity values for the A1 (CHEMBL318) and A3 receptor (CHEMBL256) on the SMILES in the 
dataset used in the previous examples.

..  code-block::
    # input is in ./data/LIGAND_RAW_small_pivot.tsv
    python -m qsprpred.predict_CLI -i LIGAND_RAW_small_pivot.tsv -sm Smiles -pr CHEMBL318 -pr CHEMBL256 -r REG -m RF

More
^^^^
The predictions can be further customized with several CLI arguments.
Here you can find a short overview.

run settings arguments
^^^^^^^^^^^^^^^^^^^^^^
As with the data preparation including `-d`, will print debug information to the log file. The random 
seed can also be set manually (although identical results are not guaranteed while keeping
the same random seed). The output file name can be set. Furthermore, the number of cpu's used for model prediction and the
gpu number for prediction with pytorch models can be set.

..  code-block::

    # Setting debug flag, random seed, output file name, number of cpu's and a specific gpu (for now multiple gpu's not possible)
        python -m qsprpred.predict_CLI -i LIGAND_RAW_small_pivot.tsv -o mypredictions -d -ran 42 -ncpu 5 -gpus [3] -sm Smiles -pr CHEMBL318 -pr CHEMBL256 -r REG -m

Model selection
^^^^^^^^^^^^^^^
You can also include multiple models predictions in the output file. By setting the model task and model types.
Make sure you have right pretrained models in the qspr/models folder.

..  code-block::

    # Making predictions with the RF and KNN classification models
    python -m qsprpred.predict_CLI -i LIGAND_RAW_small_pivot.tsv -sm Smiles -pr CHEMBL318 -pr CHEMBL256 -r CLS -m RF KNN

    
Skip SMILES preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^
By default the SMILES strings are sanitized and standardized. By including the :code:`-np` flag, this step is skipped.

..  code-block::
    
    # Do not standardize and sanitize SMILES
    python -m qsprpred.predict_CLI -i LIGAND_RAW_small_pivot.tsv -sm Smiles -pr CHEMBL318 -pr CHEMBL256 -r REG -m RF -np

