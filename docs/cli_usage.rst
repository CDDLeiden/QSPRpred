..  _cli-usage:

Command Line Interface Usage
============================

You can use the command-line interface to preprocess data and build models.
The description of the functionality can be displayed with the :code:`--help` argument,
e.g. the help message for the :code:`QSPRpred.data_CLI` script can be shown as follows:

..  code-block::

    python -m qsprpred.data_CLI --help

A simple command-line workflow to prepare your dataset and train QSPR models is given below (see :ref:`CLI Example`).

If you want more control over the inputs and outputs or want to customize QSPRpred for your purpose,
you can also use the Python API directly (see `tutorials <https://github.com/CDDLeiden/QSPRpred/tree/main/tutorials>`_).

CLI Example
***********

In this example, we will use the command line utilities of QSPRpred to train a QSAR model for the A2A and A2B receptor.
We will use the Adenosine dataset from the API tutorial. The data is available through `OneDrive <https://1drv.ms/u/s!AtzWqu0inkjX3QRxXOkTFNv7IV7u?e=PPj0O2>`_
(just unzip and place the two datasets **A2A_LIGANDS.tsv** and **AR_LIGANDS.tsv** in the 'tutorial_data' folder) or recreate the dataset yourself by running 'tutorial_data/create_tutorial_data.py'.

Input data should contain a column with SMILES sequences and at least one column with a property for modelling.
The **AR_LIGANDS.tsv** file contains a SMILES column, a column with the property **pchembl_value_mean** and a column with the property **accession** (uniprot accession numbers).
However, to create models for the A2A and A2B receptor, we need to have the data in a pivot table format, where the properties are in the columns.
To create this pivot table, we can use the pandas library in Python as follows:

..  code-block::

    import pandas as pd
    df = pd.read_csv('tutorial_data/AR_LIGANDS.tsv', sep='\t')
    df = df.pivot(index="SMILES", columns="accession", values="pchembl_value_Mean")
    df.columns.name = None
    df.reset_index(inplace=True)
    df.to_csv('tutorial_data/AR_LIGANDS_pivot.tsv', sep='\t')

It is also possible to simply run the **create_tutorial_data.py** script in the **tutorial_data** folder with the following command:

..  code-block::

    python create_tutorial_data.py -m AR_LIGANDS.tsv

This will create a pivot table with the name **AR_LIGANDS_pivot.tsv** in the **tutorial_data** folder.
Our example dataset now contains a SMILES column and two columns with the properties **P29274** (A2AR) and **P29275** (A2BR) 
(as well as a columns for A1 and A3, which we will not use in this example).

Preparing Data
--------------

.. _Data Basics:

Basics
^^^^^^
We will now use the QSPRpred :code:`QSPRpred.data_CLI` script for data preparation.
In the CLI we need to indicate which property/ies we are interested in predicting (here P29274 and P29275, the A2A and A2B receptor respectively),
this should be equal to the column names (should not contain spaces) containing the values to be predicted. 
For regression models these columns should contain numerical datapoints.
For categorical models either categorical data or numerical data can be used (the latter will be categorized based on the activity threshold).
Furthermore, we should indicate how we wish to split the data to create a train and test set.
Here we will use a random split with a test fraction of 15%. We need to calculate features that describe the molecules,
here we use Morgan fingerprints.

..  code-block::

    # input is in tutorial/data/AR_LIGANDS_pivot.tsv
    python -m qsprpred.data_CLI -i ./tutorial_data/AR_LIGANDS_pivot.tsv -o ./tutorial_output/data -pr P29274 -pr P29275 -r REG -sp random -sf 0.15 -fe Morgan

Running this command will create folders **tutorial_output/data**, with subfolders **P29274_REGRESSION** and **P29275_REGRESSION** containing the prepared data.
Each subfolder, named by the property identifiers (i.e. P29274/P29275) and the model task (REGRESSION),
will contain the following files:

+----------------------------------------------------+--------------------------------------------------------+
| File                                               | Function                                               |
+====================================================+========================================================+
|| {prefixes}_df.pkl                                 || Dataframe                                             |
|| {prefixes}_meta.json                              || Meta data, also used to instantiate a QSPRData object |
|| {prefixes}_MorganFP                               || Descriptor set folder                                 |
|| {prefixes}_MorganFP/{prefixes}_MorganFP_df.pkl    || Calculated descriptors                                |
|| {prefixes}_MorganFP/{prefixes}_MorganFP_meta.json || Meta data of the descriptor set                       |
+----------------------------------------------------+--------------------------------------------------------+

Furthermore, the command line interface will create a log file and settings file in the output folder.

+--------------------------------------------------+-------------------------------------------------------+
| File                                             | Function                                              |
+==================================================+=======================================================+
|| QSPRdata.json                                   || Command Line interface settings                      |
|| QSPRdata.log                                    || Log file                                             |
+--------------------------------------------------+-------------------------------------------------------+

.. _Data More:

More
^^^^

.. _Data Run settings arguments:

Run settings arguments
""""""""""""""""""""""
Apart from the the input file name, there are a few other base options that can be set.
Including ``-d``, will print debug information to the log file. The random
seed (``-ran``) can also be set manually, which should guarantee identical results while keeping the same random seed.
Furthermore, the number of cpu's (-ncpu) used for model training. Finally, the name of the SMILES column in your dataset
can be indicated with ``-sm`` (default SMILES).

..  code-block::

    # input is in tutorial/data/AR_LIGANDS_pivot.tsv, setting debug flag, smiles column, random seed, number of cpu's
        python -m qsprpred.data_CLI  -i ./tutorial_data/AR_LIGANDS_pivot.tsv -o ./tutorial_output/data -sm SMILES -de -ran 42 -ncpu 5 -pr P29274 -pr P29275 -r REG -sp random -sf 0.15 -fe Morgan


Transform target property
"""""""""""""""""""""""""
To apply (``-tr``) transformations to target properties, indicate this in the CLI as follows:

..  code-block::

    # Log transform data for P29274
        python -m qsprpred.data_CLI -i ./tutorial_data/AR_LIGANDS_pivot.tsv -o ./tutorial_output/data -pr P29274 -pr P29275 -tr '{"P29274":"log"}' -r REG -sp random -sf 0.15 -fe Morgan

note. on windows remove the single quotes around the dictionary and add backslashes before the double quotes, e.g. ``-tr {\"P29274\":\"log\"}``

Train-test split
""""""""""""""""
In the base example we use a random split to create the train and test set. There are several more options,
One is a scaffold split, where the data is split into a test and train set randomly but keeping molecules with the same 
(Murcko) scaffold in the same set.

..  code-block::

    # Scaffold split
        python -m qsprpred.data_CLI -i ./tutorial_data/AR_LIGANDS_pivot.tsv -o ./tutorial_output/data -pr P29274 -pr P29275 -r REG -sp scaffold -sf 0.15 -fe Morgan

Another option is the cluster split, where the data is split into a test and train set randomly but keeping molecules with the same
clusters in the same set. Here you can set the clustering method as well (``-scm``).

..  code-block::

    # Cluster split
        python -m qsprpred.data_CLI -i ./tutorial_data/AR_LIGANDS_pivot.tsv -o ./tutorial_output/data -pr P29274 -pr P29275 -r REG -sp cluster -scm MaxMin -sf 0.15 -fe Morgan

The third option is a temporal split, where a column needs to be indicated which holds information on the time each
sample was observed and split based on threshold in a column. In this example, all samples after 2015 (in column **year**)
make up the test set. NOTE: this example will not work on the example set as it does not contain a **year** column.

..  code-block::

    # Time split
        python -m qsprpred.data_CLI -i ./tutorial_data/AR_LIGANDS_pivot.tsv -o ./tutorial_output/data -pr P29274 -pr P29275 -r REG  -sp time -st 2015 -stc year -fe Morgan

Lastly, the data can be split based on a specific column in the dataset. This column has to be named **datasplit**
where the value **test** indicates the test set and the value **train** indicates the train set.
NOTE. this example will not work on the example set as it does not contain a **datasplit** column.

..  code-block::

    # Split based on a specific column
        python -m qsprpred.data_CLI -i ./tutorial_data/AR_LIGANDS_pivot.tsv -o ./tutorial_output/data -pr P29274 -pr P29275 -r REG -sp manual -sf 0.15 -fe Morgan

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
        python -m qsprpred.data_CLI -i ./tutorial_data/AR_LIGANDS_pivot.tsv -o ./tutorial_output/data -pr P29274 -pr P29275 -r CLS -sp random -sf 0.15 -fe Morgan -th '{"P29274":[6.5],"P29275":[0,4,6,12]}'

note. on windows remove the single quotes around the dictionary and add backslashes before the double quotes, e.g. ``-th {\"P29274\":[6.5],\"P29275\":[0,4,6,10]}``.

Feature calculation
"""""""""""""""""""
There are many different descriptor sets that can be calculated from the CLI,
such as Morgan fingerprints, rdkit, Mordred, Mold2 and Padel descriptors.
Check the help message for the full list of available descriptor sets.
The different descriptor sets can also be combined.
For more control over the descriptor settings use the python API.

..  code-block::

    # With Morgan, RDkit, Mordred, Mold2, PaDEL and DrugEx descriptors
        python -m qsprpred.data_CLI -i ./tutorial_data/AR_LIGANDS_pivot.tsv -o ./tutorial_output/data -pr P29274 -pr P29275 -r REG -sp random -sf 0.15 -fe Morgan RDkit Mordred Mold2 PaDEL DrugEx

Feature filtering
"""""""""""""""""
The calculated features can also be filtered. Three different filters are implemented in
QSPRpred, namely a high correlation filter, a low variance filter and the boruta filter.
The high correlation filter and low variance filter need to be set with a threshold
for filtering. The boruta filter needs a threshold for the comparison between shadow 
and real features.

..  code-block::

    # input is in ./data/LIGAND_RAW_small.tsv
       python -m qsprpred.data_CLI -i ./tutorial_data/AR_LIGANDS_pivot.tsv -o ./tutorial_output/data -pr P29274 -pr P29275 -r REG -sp random -sf 0.15 -fe Morgan -lv 0.1 -hc 0.9 -bf 90

Papyrus Low quality filter
""""""""""""""""""""""""""
Specifically for use with a dataset from the `Papyrus dataset <https://chemrxiv.org/engage/chemrxiv/article-details/617aa2467a002162403d71f0>`_,
an option is included for filtering low quality data from the dataset (All data is removed with value 'Low' in column 'Quality').
To apply this filter include ``-lq`` or ``--low_quality`` in your command.

Multitask data
""""""""""""""
Multitask modelling is possible by passing multiple properties to the ``-pr`` argument. Furthermore, missing data can be
imputed using the ``-im`` argument. You can combine any number of targets and combination of regression and classification
tasks for the data preparation, however currently the DNN models do not support multitask modelling and only the random
forest models and KNN sklearn models are supported for multitask. The multitask sklearn modelling is only possible for 
multiple regression task or multiple single class classification tasks. For multiple multi-class classification tasks or
a combination of regression and classification tasks, the multitask modelling is not supported at the moment.

..  code-block::

    # input is in ./data/parkinsons_pivot.tsv
        python -m qsprpred.data_CLI -i ./tutorial_data/AR_LIGANDS_pivot.tsv -o ./tutorial_output/data -pr P29274 P29275 -r REG -sp random -sf 0.15 -fe Morgan -im '{"P29274":"mean", "P29275":"median"}'

Note. on windows remove the single quotes around the dictionary and add backslashes before the double quotes, e.g. -im {\"P29274\":\"mean\",\"P29275\":\"median\"}

Model Training
--------------

.. _Model Basics:

Basics
^^^^^^

Finally, we need to indicate what models we want to train and which steps to take in the training.
In this example, we will build regression random forest models through passing the prepared regression datasets files
**P29274_REGRESSION** and **P29275_REGRESSION** to the ``-dp`` argument. If you wish to train classification models, you
can pass the classification datasets **P29274_SINGLECLASS** and **P29275_MULTICLASS** to the ``-dp`` argument
(or any combination thereof). The model type is set with ``-m``. 
We will also evaluate the model through cross-validation (``-me``) and train the model on all data to save for further use (``-s``).

..  code-block::

    # Using the prepared datasets P29274_REGRESSION and P29275_REGRESSION
        python -m qsprpred.model_CLI -dp ./tutorial_output/data/P29274_REGRESSION/P29274_REGRESSION_meta.json ./tutorial_output/data/P29275_REGRESSION/P29275_REGRESSION_meta.json -o ./tutorial_output/models -mt RF -me -s

This will create a folder **tutorial_output/models** containing the trained models.
Each subfolder, named by the model type (RF) and the dataset name (P29274_REGRESSION/P29275_REGRESSION),
will contain the following files:

+--------------------------------------------------+---------------------------------------------------------+
| File                                             | Function                                                |
+==================================================+=========================================================+
|| {prefixes}.json                                 || Model file                                             |
|| {prefixes}_meta.json                            || Meta data, also used to instantiate a QSPRModel object |
|| {prefixes}_cv.tsv                               || Cross-validation predictions                           |
|| {prefixes}_ind.tsv                              || Test set predictions                                   |
+--------------------------------------------------+---------------------------------------------------------+

Furthermore, the command line interface will create a log file and settings file in the output folder.

+--------------------------------------------------+-------------------------------------------------------+
| File                                             | Function                                              |
+==================================================+=======================================================+
|| QSPRmodel.json                                  || Command Line interface settings                      |
|| QSPRmodel.log                                   || Log file                                             |
+--------------------------------------------------+-------------------------------------------------------+

.. _Model More:

More
^^^^
The model training can be further customized with several CLI arguments.
For more control over the model training settings use the python API.
Here you can find a short overview.

.. _Model Run settings arguments:

Run settings arguments
""""""""""""""""""""""
As with the data preparation including ``-de``, will print debug information to the log file. The random 
seed can also be set manually (although identical results are not guaranteed while keeping
the same random seed). Furthermore, the number of cpu's used for model training and the
gpu number for training pytorch models can be set.

..  code-block::

    # Setting debug flag, random seed, number of cpu's and a specific gpu (for now multiple gpu's not possible)
        python -m qsprpred.model_CLI -de -ran 42 -ncpu 5 -gpus [3] -dp ./tutorial_output/data/P29274_REGRESSION/P29274_REGRESSION_meta.json ./tutorial_output/data/P29275_REGRESSION/P29275_REGRESSION_meta.json -o ./tutorial_output/models -mt RF -me -s

model types
"""""""""""
You also need to indicate which models you want to run, out of the following model types:
'RF' (Random Forest), 'XGB' (XGboost), 'SVM' (Support Vector Machine), 'PLS' (partial least squares regression),
'KNN' (k-nearest neighbours), NB' (Naive Bayes) and/or 'DNN' (pytorch fully connected neural net).
The default is to run all the different model types.

..  code-block::

    # Training a RF, SVM and PLS model
        python -m qsprpred.model_CLI -dp ./tutorial_output/data/P29274_REGRESSION/P29274_REGRESSION_meta.json ./tutorial_output/data/P29275_REGRESSION/P29275_REGRESSION_meta.json -o ./tutorial_output/models -me -s -mt RF SVM PLS

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
        python -m qsprpred.model_CLI -dp ./tutorial_output/data/P29274_REGRESSION/P29274_REGRESSION_meta.json ./tutorial_output/data/P29275_REGRESSION/P29275_REGRESSION_meta.json -o ./tutorial_output/models -mt RF KNN -me -s -p ./tutorial_output/myparams

Specifically for the training of the DNN model, you can set the tolerance and the patience from the CLI.
Tolerance gives the mimimum decrease in loss needed to count as an improvement and 
patience is the number of training epochs without improvement in loss to stop the training.

..  code-block::

    # Setting the tolerance and patience for training a DNN model
        python -m qsprpred.model_CLI -dp ./tutorial_output/data/P29274_REGRESSION/P29274_REGRESSION_meta.json ./tutorial_output/data/P29275_REGRESSION/P29275_REGRESSION_meta.json -o ./tutorial_output/models -mt DNN -me -s -tol 0.02 -pat 100

Hyperparameter optimization
"""""""""""""""""""""""""""
In addition to setting model parameters manually, a hyperparameter search can be performed.
In QSPRpred, two methods of hyperparameter optimization are implemented: grid search and 
bayesian optimization. For baysian optimization also give the number of trials.
The search space needs to be set using a json file.
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
        python -m qsprpred.model_CLI -dp ./tutorial_output/data/P29274_REGRESSION/P29274_REGRESSION_meta.json ./tutorial_output/data/P29275_REGRESSION/P29275_REGRESSION_meta.json -o ./tutorial_output/models -mt RF -me -s -o bayes -nt 5 -ss ./tutorial_output/mysearchspace.json -me -s

Multitask modelling
"""""""""""""""""""
Multitask modelling is also possible. This means that the models are trained on multiple targets at once.
The modelling arguments are the same as for single task modelling, you just need to specifiy a multitask
dataset data prefix (see multitask data preparation).

Prediction
-----------
Furthermore, trained QSPRpred models can be used to predict values from SMILES from the command line interface :code:`predict_CLI.py`.

.. _Prediction Basics:

Basics
^^^^^^
Here we will predict activity values for the A1 (P29274) and A3 receptor (P29275) on the SMILES in the 
dataset used in the previous examples using the models from the previous examples. The input ``-i`` here is the 
set of SMILES for which we want to predict activity values. The argument ``-mp``, is the paths to the meta files of the 
models we want to use for prediction relative to the base-directory subfolder qspr/models.

..  code-block::
    
    # Making predictions for the A2A and A2B receptor
    python -m qsprpred.predict_CLI -i ./tutorial_data/AR_LIGANDS_pivot.tsv -o ./tutorial_output/predictions/AR_LIGANDS_preds.tsv -mp ./tutorial_output/models/RF_P29274_REGRESSION/RF_P29274_REGRESSION_meta.json ./tutorial_output/models/RF_P29275_REGRESSION/RF_P29275_REGRESSION_meta.json

.. _Prediction More:

More
^^^^
The predictions can be further customized with several CLI arguments.
Here you can find a short overview.

.. _Prediction Run settings arguments:

Run settings arguments
""""""""""""""""""""""
As with the data preparation including ``-de``, will print debug information to the log file. The random 
seed can also be set manually. Furthermore, the number of cpu's used for model prediction and the
gpu number for prediction with pytorch models can be set.

..  code-block::

    # Setting debug flag, random seed, output file name, number of cpu's and a specific gpu (for now multiple gpu's not possible)
        python -m qsprpred.predict_CLI -i ./tutorial_data/AR_LIGANDS_pivot.tsv -o ./tutorial_output/predictions/AR_LIGANDS_preds.tsv -mp ./tutorial_output/models/RF_P29274_REGRESSION/RF_P29274_REGRESSION_meta.json ./tutorial_output/models/RF_P29275_REGRESSION/RF_P29275_REGRESSION_meta.json -de -ran 42 -ncpu 5 -gpus [3]

    
Adding probability predictions
""""""""""""""""""""""""""""""
When using a classification model, the probability of the predicted class can be added to the output file using the ``-pr`` flag.

..  code-block::
    
    # Adding probability predictions
    python -m qsprpred.predict_CLI -i ./tutorial_data/AR_LIGANDS_pivot.tsv -o ./tutorial_output/predictions/AR_LIGANDS_preds.tsv -mp ./tutorial_output/models/RF_P29274_SINGLECLASS/RF_P29274_SINGLECLASS_meta.json ./tutorial_output/models/RF_P29275_MULTICLASS/RF_P29275_MULTICLASS_meta.json -pr

