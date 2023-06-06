..  _usage:

Usage
=====

You can use the command-line interface to preprocess data and build models.
The description of the functionality can be displayed with the :code:`--help` argument,
e.g. the help message for the :code:`QSPRpred.data_CLI` script can be shown as follows:

..  code-block::

    python -m qsprpred.data_CLI --help

A simple command-line workflow to prepare your dataset and train QSPR models is given below (see :ref:`cli-example`).

If you want more control over the inputs and outputs or want to customize QSPRpred for your purpose,
you can also use the Python API directly (see `source code <https://github.com/CDDLeiden/QSPRpred/tree/main/tutorial>`).
Here you can find a tutorial with a Jupyter notebook illustrating some common use cases in the project source code.
Make sure to download the tutorial folder to follow the examples in this CLI tutorial.

..  _cli-example:

CLI Example
===========

In this example, we will use the command line utilities of QSPRpred to train a QSAR model for the GABAA receptor and the 
Glutamate receptor NMDA. We will use the parkinsons dataset from the API tutorial. 
Make sure to first run the Parkinsons function in the tutorial/datasets.py script (see dataset preparation in the API tutorial)
to create the tutorial/data/parkinsons_pivot.tsv file.
QSPRpred assumes that all input data are saved in a subfolder data of the base directory..
Input data should contain a column with SMILES sequences and at least one column with a property for modelling.
Therefore, our example dataset 'parkinsons_pivot.tsv' is located in a subfolder 'data' of the tutorial directory.

Preparing Data
--------------

Basics
^^^^^^
We will now use the QSPRpred :code:`QSPRpred.data_CLI` script for data preparation.
In the CLI we need to indicate which property/ies we are interested in predicting (here GABAAalpha and NMDA),
this should be equal to the column names (should not contain spaces) containing the values to be predicted. 
For regression models these columns should contain numerical datapoints.
For categorical models either categorical data or numerical data can be used (the latter will be categorized based on the activity threshold).
Furthermore, we should indicate how we wish to split the data to create a train and test set.
Here we will use a random split with a test fraction of 15%. We need to calculate features that describe the molecules,
here we use morgan fingerprints.

..  code-block::

    # input is in ./data/parkinsons_pivot.tsv
        python -m qsprpred.data_CLI -i parkinsons_pivot.tsv -pr GABAAalpha -pr NMDA -r REG -sp random -sf 0.15 -fe Morgan

Running this command will create a subfolder 'qspr'. The qspr/data folder should contain 4 files for
each property (8 in total), prefixed by the property identifiers (i.e. GABAAalpha/NMDA), the model task (REGRESSION),
e.g. GABAAalpha_REGRESSION. As well as an log file and a run settings file.

+--------------------------------------------------+-------------------------------------------------------+
| File                                             | Function                                              |
+==================================================+=======================================================+
|| {prefixes}_df.pkl                               || Dataframe                                            |
|| {prefixes}_feature_calculators.json             || re-instantiate feature calculator                    |
|| {prefixes}_feature_standardizer.json            || re-instantiate feature standardizer                  |
|| {prefixes}_meta.json                            || Meta data, also used to instantiate QSAR data object |
|| QSPRdata.json                                   || Command Line interface settings                      |
|| QSPRdata.log                                    || Log file                                             |
+--------------------------------------------------+-------------------------------------------------------+


More
^^^^
Run settings arguments
^^^^^^^^^^^^^^^^^^^^^^^
Apart from the the input file name, there are a few other base options that can be set.
The base-directory can be specified using `-b`. Including `-d`, will print debug information to the log file. The random
seed (-ran) can also be set manually (although identical results are not guaranteed while keeping the same random seed).
Furthermore, the number of cpu's (-ncpu) used for model training. Finally, the name of the smilescolumn in your dataset
can be indicated with `-sm` (default SMILES).

..  code-block::

    # input is in tutorial/data/parkinsons_pivot.tsv, setting debug flag, smiles column, random seed, number of cpu's
        python -m qsprpred.data_CLI -b tutorial -i parkinsons_pivot.tsv -sm SMILES -de -ran 42 -ncpu 5 -pr GABAAalpha -pr NMDA -r REG -sp random -sf 0.15 -fe Morgan


Log-transform data
""""""""""""""""""
To log (-lt) transform data specific properties, indicate this in the CLI as follows:

..  code-block::

    # Log transform data for GABAAalpha
        python -m qsprpred.data_CLI -i parkinsons_pivot.tsv -pr GABAAalpha -pr NMDA -lt '{"GABAAalpha":true,"NMDA":false}' -r REG -sp random -sf 0.15 -fe Morgan

Train test split
""""""""""""""""
In the base example we use a random split to create the train and test set. There are two more options,
namely a scaffold split, where the data is split into a test and train set randomly but keeping molecules with the same 
(Murcko) scaffold in the same set.

..  code-block::

    # Scaffold split
        python -m qsprpred.data_CLI -i parkinsons_pivot.tsv -pr GABAAalpha -pr NMDA -r REG -sp scaffold -sf 0.15 -fe Morgan

The third option is a temporal split, where a column needs to be indicated which holds information on the time each
sample was observed and split based on threshold in a column. In this example, all samples after 2015 (in column 'year')
make up the test set. NOTE: this example will not work on the example set as it does not contain a 'year' column.

..  code-block::

    # Time split
        python -m qsprpred.data_CLI -i parkinsons_pivot.tsv -pr GABAAalpha -pr NMDA -r REG  -sp time -st 2015 -stc year -fe Morgan


Data for classification models
""""""""""""""""""""""""""""""
You can set whether to prepare data for regression, classification or both.
The default setting is to run both, but you can run either by setting the
regression argument to true/REG for regression or false/CLS for classification.
When using classification, the threshold(s) for each property (that has not been preclassified) need to be included.
If the data is already preclassified, the threshold has to be set to 'precomputed'.
This is set using a dictionary. In case of multi-class classification the bounderies of
the bins need to be given. For binary classification only give 1 threshold per property.

..  code-block::

    # Classification and regression
        python -m qsprpred.data_CLI -i parkinsons_pivot.tsv -pr GABAAalpha -pr NMDA -r CLS -sp random -sf 0.15 -fe Morgan -th '{"GABAAalpha":[6.5],"NMDA":[0, 4, 6, 10]}'

Feature calculation
"""""""""""""""""""
There are seven different descriptor sets that can be calculated at the moment,
namely Morgan fingerprints, rdkit, Mordred, Mold2 and Padel descriptors, the
physicochemical properties used in the QSAR models in the DrugEx papers, and
the SMILES based signatures of extended valence. They can also
be combined. For more control over the descriptorcalculator settings use the python API.

..  code-block::

    # With Morgan, RDkit, Mordred, Mold2, PaDEL and DrugEx descriptors
        python -m qsprpred.data_CLI -i parkinsons_pivot.tsv -pr GABAAalpha -pr NMDA -r REG -sp random -sf 0.15 -fe Morgan RDkit Mordred Mold2 PaDEL DrugEx

Feature filtering
"""""""""""""""""
The calculated features can also be filtered. Three different filters are implemented in
QSPRpred, namely a high correlation filter, a low variance filter and the boruta filter.
The high correlation filter and low variance filter need to be set with a threshold
for filtering.

..  code-block::

    # input is in ./data/LIGAND_RAW_small.tsv
       python -m qsprpred.data_CLI -i parkinsons_pivot.tsv -pr GABAAalpha -pr NMDA -r REG -sp random -sf 0.15 -fe Morgan -lv 0.1 -hc 0.9 -bf

Papyrus Low quality filter
""""""""""""""""""""""""""
Specifically for use with a dataset from the `Papyrus dataset <https://chemrxiv.org/engage/chemrxiv/article-details/617aa2467a002162403d71f0>`,
an option is included for filtering low quality data from the dataset (All data is removed with value 'Low' in column 'Quality').
To apply this filter include `-lq` or `--low_quality` in your command.

Multitask data
""""""""""""""
Multitask modelling is possible by passing multiple properties to the `-pr` argument. Furthermore, missing data can be
imputed using the `-im` argument. You can combine any number of targets and combination of regression and classification
tasks for the data preparation, however currently the DNN models do not support multitask modelling and only the random
forest models and knn sklearn models are supported for multitask. The multitask sklearn modelling is only possible for 
multiple regression task or multiple single class classification tasks. For multiple multi-class classification tasks or
a combination of regression and classification tasks, the multitask modelling is not supported at the moment.

..  code-block::

    # input is in ./data/parkinsons_pivot.tsv
        python -m qsprpred.data_CLI -i parkinsons_pivot.tsv -pr GABAAalpha NMDA -r REG -sp random -sf 0.15 -fe Morgan -im mean

Model Training
--------------

Basics
^^^^^^

Finally, we need to indicate what models we want to train and which steps to take in the training.
In this example, we will build regression random forest models through passing the prepared regression datasets prefixes
`GABAAalpha_REGRESSION` and `NMDA_REGRESSION` to the `-dp` argument. If you wish to train classification models, you
can pass the classification datasets `GABAAalpha_CLASSIFICATION` and `NMDA_CLASSIFICATION` to the `-dp` argument
(or any combination thereof). The model type is set with `-m`. 
We will also evaluate the model through cross-validation (-me) and train the model on all data to save for further use (-s).

..  code-block::

    # input is in ./data/parkinsons_pivot.tsv
        python -m qsprpred.model_CLI -dp GABAAalpha_REGRESSION NMDA_REGRESSION -mt RF -me -s

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
        python -m qsprpred.model_CLI -de -ran 42 -ncpu 5 -gpus [3] -dp GABAAalpha_REGRESSION NMDA_REGRESSION -mt RF -me -s

model types
"""""""""""
You also need to indicate which models you want to run, out of the following model types:
'RF' (Random Forest), 'XGB' (XGboost), 'SVM' (Support Vector Machine), 'PLS' (partial least squares regression),
'KNN' (k-nearest neighbours), NB' (Naive Bayes) and/or 'DNN' (pytorch fully connected neural net).
The default is to run all the different model types.

..  code-block::

    # Training a RF, SVM and PLS model
        python -m qsprpred.model_CLI -dp GABAAalpha_REGRESSION NMDA_REGRESSION -me -s -mt RF SVM PLS

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
        python -m qsprpred.model_CLI -dp GABAAalpha_REGRESSION NMDA_REGRESSION -mt RF KNN -me -s -p myparams

Specifically for the training of the DNN model, you can set the tolerance and the patience from the CLI.
Tolerance gives the mimimum decrease in loss needed to count as an improvement and 
patience is the number of training epochs without improvement in loss to stop the training.

..  code-block::

    # Setting the tolerance and patience for training a DNN model
        python -m qsprpred.model_CLI -dp GABAAalpha_REGRESSION NMDA_REGRESSION -mt DNN -me -s -tol 0.02 -pat 100

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
        python -m qsprpred.model_CLI -dp GABAAalpha_REGRESSION NMDA_REGRESSION -mt RF -me -s -o bayes -nt 50 -ss mysearchspace -me -s

Multitask modelling
"""""""""""""""""""
Multitask modelling is also possible. This means that the models are trained on multiple targets at once.
The modelling arguments are the same as for single task modelling, you just need to specifiy the a multitask
dataset data prefix (see multitask data preparation).


Prediction
-----------
Furthermore, trained QSPRpred models can be used to predict values from SMILES from the command line interface :code:`predict_CLI.py`.

Basics
^^^^^^
Here we will predict activity values for the A1 (GABAAalpha) and A3 receptor (NMDA) on the SMILES in the 
dataset used in the previous examples using the models from the previous examples. The input `-i` here is the 
set of SMILES for which we want to predict activity values. The argument `-mp`, is the paths to the meta files of the 
models we want to use for prediction relative to the base-directory subfolder qspr/models.

..  code-block::
    
    # input is in ./data/parkinsons_pivot.tsv
    python -m qsprpred.predict_CLI -i parkinsons_pivot.tsv -mp RF_GABAAalpha_REGRESSION/RF_GABAAalpha_REGRESSION_meta.json RF_NMDA_REGRESSION/RF_NMDA_REGRESSION_meta.json

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
        python -m qsprpred.predict_CLI -i parkinsons_pivot.tsv -mp RF_GABAAalpha_REGRESSION/RF_GABAAalpha_REGRESSION_meta.json RF_NMDA_REGRESSION/RF_NMDA_REGRESSION_meta.json -o mypredictions -de -ran 42 -ncpu 5 -gpus [3]

    
Adding probability predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When using a classification model, the probability of the predicted class can be added to the output file using the `-pr` flag.

..  code-block::
    
    # Do not standardize and sanitize SMILES
    python -m qsprpred.predict_CLI -i parkinsons_pivot.tsv -mp RF_GABAAalpha_SINGLECLASS/RF_GABAAalpha_SINGLECLASS_meta.json RF_NMDA_MULTICLASS/RF_NMDA_MULTICLASS_meta.json -pr

