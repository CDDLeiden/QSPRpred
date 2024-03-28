import logging
import os
from copy import deepcopy
from os.path import exists
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .path_mixins import ModelDataSetsPathMixIn
from ... import TargetTasks
from ...data import QSPRDataset
from ...data.descriptors.sets import DescriptorSet
from ...data.processing.feature_standardizers import SKLearnStandardizer
from ...models import (
    QSPRModel,
    OptunaOptimization,
    CrossValAssessor,
    EarlyStoppingMode,
    SklearnMetrics,
    GridSearchOptimization,
    TestSetAssessor,
    HyperparameterOptimizationMonitor,
    AssessorMonitor,
    FitMonitor,
    BaseMonitor,
    FileMonitor,
    HyperparameterOptimization,
)
from ...models.monitors import ListMonitor
from ...tasks import TargetProperty


class DescriptorCheckMixIn:
    """Mixin class for common descriptor checks."""

    def checkFeatures(self, ds: QSPRDataset, expected_length: int):
        """Check if the feature names and the feature matrix of a data set is consistent
        with expected number of variables.

        Args:
            ds (QSPRDataset): The data set to check.
            expected_length (int): The expected number of features.

        Raises:
            AssertionError: If the feature names or the feature matrix is not consistent
        """
        self.assertEqual(len(ds.featureNames), expected_length)
        self.assertEqual(len(ds.getFeatureNames()), expected_length)
        if expected_length > 0:
            features = ds.getFeatures(concat=True)
        else:
            self.assertRaises(ValueError, ds.getFeatures, concat=True)
            features = pd.concat([ds.X, ds.X_ind])
        self.assertEqual(features.shape[0], len(ds))
        self.assertEqual(features.shape[1], expected_length)
        self.assertEqual(ds.X.shape[1], expected_length)
        self.assertEqual(ds.X_ind.shape[1], expected_length)
        if expected_length > 0:
            for fold in ds.iterFolds(split=KFold(n_splits=5)):
                self.assertIsInstance(fold, tuple)
                self.assertEqual(fold[0].shape[1], expected_length)
                self.assertEqual(fold[1].shape[1], expected_length)
        else:
            self.assertRaises(
                ValueError, lambda: list(ds.iterFolds(split=KFold(n_splits=5)))
            )

        # check if outliers are dropped
        if "TestOutlier" in ds.df.columns:
            num_dropped = ds.df.TestOutlier.sum()
            # expected number of samples is the total number of samples minus the number
            # of samples in the training set, minus the number of dropped
            expected_num_samples = len(ds) - (len(ds.X)) - num_dropped
            X, X_ind = ds.getFeatures(concat=False)
            self.assertEqual(X_ind.shape[0], expected_num_samples)

    def checkDescriptors(
        self, dataset: QSPRDataset, target_props: list[dict | TargetProperty]
    ):
        """
        Check if information about descriptors is consistent in the data set. Checks
        if calculators are consistent with the descriptors contained in the data set.
        This is tested also before and after serialization.

        Args:
            dataset (QSPRDataset): The data set to check.
            target_props (List of dicts or TargetProperty): list of target properties

        Raises:
            AssertionError: If the consistency check fails.

        """

        # test some basic consistency rules on the resulting features
        expected_length = 0
        for calc in dataset.descriptorSets:
            expected_length += len(calc.descriptors)
        self.checkFeatures(dataset, expected_length)
        # save to file, check if it can be loaded, and if the features are consistent
        dataset.save()
        ds_loaded = dataset.__class__.fromFile(dataset.metaFile)
        self.assertEqual(ds_loaded.nJobs, dataset.nJobs)
        self.assertEqual(ds_loaded.chunkSize, dataset.chunkSize)
        self.assertEqual(ds_loaded.randomState, dataset.randomState)
        for ds_loaded_prop, target_prop in zip(
            ds_loaded.targetProperties, target_props
        ):
            if ds_loaded_prop.task.isClassification():
                self.assertEqual(ds_loaded_prop.name, target_prop["name"])
                self.assertEqual(ds_loaded_prop.task, target_prop["task"])
        self.assertTrue(ds_loaded.descriptorSets)
        for calc in ds_loaded.descriptors:
            calc = calc.calculator
            self.assertTrue(isinstance(calc, DescriptorSet))
        self.checkFeatures(dataset, expected_length)


class DataPrepCheckMixIn(DescriptorCheckMixIn):
    """Mixin for testing data preparation."""

    def checkPrep(
        self,
        dataset,
        feature_calculators,
        split,
        feature_standardizer,
        feature_filter,
        data_filter,
        applicability_domain,
        expected_target_props,
    ):
        """Check the consistency of the dataset after preparation."""
        name = dataset.name
        # if a split needs a dataset, give it one
        if split and hasattr(split, "setDataSet"):
            split.setDataSet(None)
            self.assertRaises(ValueError, split.getDataSet)
            split.setDataSet(dataset)
            self.assertEquals(dataset, split.getDataSet())

        # prepare the dataset and check consistency
        dataset.prepareDataset(
            feature_calculators=feature_calculators,
            split=split if split else None,
            feature_standardizer=feature_standardizer if feature_standardizer else None,
            feature_filters=[feature_filter] if feature_filter else None,
            data_filters=[data_filter] if data_filter else None,
            applicability_domain=applicability_domain,
            drop_outliers=True if applicability_domain is not None else False,
        )
        expected_feature_count = len(dataset.featureNames)
        original_features = dataset.featureNames
        train, test = dataset.getFeatures()
        self.checkFeatures(dataset, expected_feature_count)
        # save the dataset
        dataset.save()
        # reload the dataset and check consistency again
        dataset = dataset.__class__.fromFile(dataset.metaFile)
        self.assertEqual(dataset.name, name)
        self.assertEqual(dataset.targetProperties[0].task, TargetTasks.REGRESSION)
        for idx, prop in enumerate(expected_target_props):
            self.assertEqual(dataset.targetProperties[idx].name, prop)
        for calc in dataset.descriptors:
            calc = calc.calculator
            self.assertIsInstance(calc, DescriptorSet)
        if feature_standardizer is not None:
            self.assertIsInstance(dataset.featureStandardizer, SKLearnStandardizer)
        else:
            self.assertIsNone(dataset.featureStandardizer)
        self.checkFeatures(dataset, expected_feature_count)
        # verify prep results are the same after reloading
        dataset.prepareDataset(
            feature_calculators=feature_calculators,
            split=split if split else None,
            feature_standardizer=feature_standardizer if feature_standardizer else None,
            feature_filters=[feature_filter] if feature_filter else None,
            data_filters=[data_filter] if data_filter else None,
            applicability_domain=applicability_domain,
            drop_outliers=True if applicability_domain is not None else False,
        )
        self.checkFeatures(dataset, expected_feature_count)
        self.assertListEqual(sorted(dataset.featureNames), sorted(original_features))
        train2, test2 = dataset.getFeatures()
        self.assertTrue(train.index.equals(train2.index))
        self.assertTrue(test.index.equals(test2.index))


class DescriptorInDataCheckMixIn(DescriptorCheckMixIn):
    """Mixin for testing descriptor sets in data sets."""

    @staticmethod
    def getDatSetName(desc_set, target_props):
        """Get a unique name for a data set."""
        target_props_id = [
            f"{target_prop['name']}_{target_prop['task']}"
            for target_prop in target_props
        ]
        return f"{desc_set}_{target_props_id}"

    def checkDataSetContainsDescriptorSet(
        self, dataset, desc_set, prep_combo, target_props
    ):
        """Check if a descriptor set is in a data set."""
        # run the preparation
        logging.debug(f"Testing descriptor set: {desc_set} in data set: {dataset.name}")
        preparation = {}
        preparation.update(prep_combo)
        preparation["feature_calculators"] = [desc_set]
        dataset.prepareDataset(**preparation)
        # test consistency
        self.checkDescriptors(dataset, target_props)


class ModelCheckMixIn:
    """This class holds the tests for the QSPRmodel class."""

    @property
    def gridFile(self):
        return f"{os.path.dirname(__file__)}/test_files/search_space_test.json"

    def getParamGrid(self, model: QSPRModel, grid: str) -> dict:
        """Get the parameter grid for a model.

        Args:
            model (QSPRModel): The model to get the parameter grid for.
            grid (str): The grid type to get the parameter grid for.

        Returns:
            dict: The parameter grid.
        """

        mname = model.name.split("_")[0]
        grid_params = model.__class__.loadParamsGrid(self.gridFile, grid, mname)
        return grid_params[grid_params[:, 0] == mname, 1][0]

    def checkOptimization(
        self, model: QSPRModel, ds: QSPRDataset, optimizer: HyperparameterOptimization
    ):
        model_path, est_path = model.save(save_estimator=True)
        # get last modified time stamp of the model file
        model_last_modified = os.path.getmtime(est_path)
        best_params = optimizer.optimize(model, ds)
        for param in best_params:
            self.assertEqual(best_params[param], model.parameters[param])
        new_time_modified = os.path.getmtime(est_path)
        self.assertTrue(model_last_modified < new_time_modified)
        optimizer.optimize(model, ds, refit_optimal=True)
        model_last_modified = new_time_modified
        new_time_modified = os.path.getmtime(est_path)
        self.assertTrue(model_last_modified < new_time_modified)
        model_new = model.__class__.fromFile(model.metaFile)
        for param in model.parameters:
            self.assertEqual(model_new.parameters[param], model.parameters[param])

    def fitTest(self, model: QSPRModel, ds: QSPRDataset):
        """Test model fitting, optimization and evaluation.

        Args:
            model (QSPRModel): The model to test.
            ds (QSPRDataset): The dataset to use for testing.
        """
        # perform bayes optimization
        model.initFromDataset(ds)
        score_func = "r2" if model.task.isRegression() else "roc_auc_ovr"
        search_space_bs = self.getParamGrid(model, "bayes")
        bayesoptimizer = OptunaOptimization(
            param_grid=search_space_bs,
            n_trials=1,
            model_assessor=CrossValAssessor(
                scoring=score_func, mode=EarlyStoppingMode.NOT_RECORDING
            ),
        )
        self.checkOptimization(model, ds, bayesoptimizer)
        model.cleanFiles()
        # perform grid search
        search_space_gs = self.getParamGrid(model, "grid")
        if model.task.isClassification():
            score_func = SklearnMetrics("accuracy")
        gridsearcher = GridSearchOptimization(
            param_grid=search_space_gs,
            score_aggregation=np.median,
            model_assessor=TestSetAssessor(
                scoring=score_func,
                use_proba=False,
                mode=EarlyStoppingMode.NOT_RECORDING,
            ),
        )
        self.checkOptimization(model, ds, gridsearcher)
        model.cleanFiles()
        # perform crossvalidation
        score_func = "r2" if model.task.isRegression() else "roc_auc_ovr"
        n_folds = 5
        scores = CrossValAssessor(
            mode=EarlyStoppingMode.RECORDING,
            scoring=score_func,
            split_multitask_scores=model.isMultiTask,
            split=KFold(n_splits=n_folds, shuffle=True, random_state=model.randomState),
        )(model, ds)
        if model.isMultiTask:
            self.assertEqual(scores.shape, (n_folds, len(model.targetProperties)))
        scores = TestSetAssessor(
            mode=EarlyStoppingMode.NOT_RECORDING,
            scoring=score_func,
            split_multitask_scores=model.isMultiTask,
        )(model, ds)
        if model.isMultiTask:
            self.assertEqual(scores.shape, (len(model.targetProperties),))
        self.assertTrue(exists(f"{model.outDir}/{model.name}.ind.tsv"))
        self.assertTrue(exists(f"{model.outDir}/{model.name}.cv.tsv"))
        # train the model on all data
        path = model.fitDataset(ds)
        self.assertTrue(exists(path))
        self.assertTrue(exists(model.metaFile))
        self.assertEqual(path, model.metaFile)

    def predictorTest(
        self,
        model: QSPRModel,
        dataset: QSPRDataset,
        comparison_model: QSPRModel | None = None,
        expect_equal_result=True,
        **pred_kwargs,
    ):
        """Test model predictions.

        Checks if the shape of the predictions is as expected and if the predictions
        of the predictMols function are consistent with the predictions of the
        predict/predictProba functions. Also checks if the predictions of the model are
        the same as the predictions of the comparison model if given.

        Args:
            model (QSPRModel): The model to make predictions with.
            dataset (QSPRDataset): The dataset to make predictions for.
            comparison_model (QSPRModel): another model to compare the predictions with.
            expect_equal_result (bool): Whether the expected result should be equal or
                not equal to the predictions of the comparison model.
            **pred_kwargs:
                Extra keyword arguments to pass to the predictor's `predictMols` method.
        """

        # define checks of the shape of the predictions
        def check_shape(predictions, model, num_smiles, use_probas):
            if model.task.isClassification() and use_probas:
                # check predictions are a list of arrays of shape (n_smiles, n_classes)
                self.assertEqual(len(predictions), len(model.targetProperties))
                for i in range(len(model.targetProperties)):
                    self.assertEqual(
                        predictions[i].shape,
                        (num_smiles, model.targetProperties[i].nClasses),
                    )
            else:
                # check predictions are an array of shape (n_smiles, n_targets)
                self.assertEqual(
                    predictions.shape,
                    (num_smiles, len(model.targetProperties)),
                )

        # define check for comparing predictions with expected result
        def check_predictions(predictions, expected_result, expect_equal_result):
            # check if predictions are almost equal to expected result (rtol=1e-5)
            check_outcome = self.assertTrue if expect_equal_result else self.assertFalse
            if isinstance(expected_result, list):
                for i in range(len(expected_result)):
                    check_outcome(np.allclose(predictions[i], expected_result[i]))
            else:
                check_outcome(np.allclose(predictions, expected_result))

        # Check if the predictMols function gives the same result as the
        # predict/predictProba function
        # get the expected result from the basic predict function
        features = dataset.getFeatures(
            concat=True, ordered=True, refit_standardizer=False
        )
        expected_result = model.predict(features)
        # make predictions with the predictMols function and check with previous result
        smiles = list(dataset.smiles)
        num_smiles = len(smiles)
        predictions = model.predictMols(smiles, use_probas=False, **pred_kwargs)
        check_shape(predictions, model, num_smiles, use_probas=False)
        check_predictions(predictions, expected_result, True)
        # do the same for the predictProba function
        predictions_proba = None
        if model.task.isClassification():
            expected_result_proba = model.predictProba(features)
            predictions_proba = model.predictMols(
                smiles, use_probas=True, **pred_kwargs
            )
            check_shape(predictions_proba, model, len(smiles), use_probas=True)
            check_predictions(predictions_proba, expected_result_proba, True)
        # check if the predictions are (not) the same as of the comparison model
        if comparison_model is not None:
            predictions_comparison = comparison_model.predictMols(
                smiles, use_probas=False, **pred_kwargs
            )
            check_predictions(predictions, predictions_comparison, expect_equal_result)
            if predictions_proba is not None:
                predictions_comparison_proba = comparison_model.predictMols(
                    smiles, use_probas=True, **pred_kwargs
                )
                check_predictions(
                    predictions_proba, predictions_comparison_proba, expect_equal_result
                )


class MonitorsCheckMixIn(ModelDataSetsPathMixIn, ModelCheckMixIn):
    def trainModelWithMonitoring(
        self,
        model: QSPRModel,
        ds: QSPRDataset,
        hyperparam_monitor: HyperparameterOptimizationMonitor,
        crossval_monitor: AssessorMonitor,
        test_monitor: AssessorMonitor,
        fit_monitor: FitMonitor,
    ) -> (
        HyperparameterOptimizationMonitor,
        AssessorMonitor,
        AssessorMonitor,
        FitMonitor,
    ):
        score_func = (
            "r2" if ds.targetProperties[0].task.isRegression() else "roc_auc_ovr"
        )
        search_space_gs = self.getParamGrid(model, "grid")
        gridsearcher = GridSearchOptimization(
            param_grid=search_space_gs,
            model_assessor=CrossValAssessor(
                scoring=score_func,
                mode=EarlyStoppingMode.NOT_RECORDING,
            ),
            monitor=hyperparam_monitor,
        )
        best_params = gridsearcher.optimize(model, ds)
        model.setParams(best_params)
        model.save()
        # perform crossvalidation
        CrossValAssessor(
            mode=EarlyStoppingMode.RECORDING,
            scoring=score_func,
            monitor=crossval_monitor,
        )(model, ds)
        TestSetAssessor(
            mode=EarlyStoppingMode.NOT_RECORDING,
            scoring=score_func,
            monitor=test_monitor,
        )(model, ds)
        # train the model on all data
        model.fitDataset(ds, monitor=fit_monitor)
        return hyperparam_monitor, crossval_monitor, test_monitor, fit_monitor

    def baseMonitorTest(
        self,
        monitor: BaseMonitor,
        monitor_type: Literal["hyperparam", "crossval", "test", "fit"],
        neural_net: bool,
    ):
        """Test the base monitor."""

        def check_fit_empty(monitor):
            self.assertEqual(len(monitor.fitLog), 0)
            self.assertEqual(len(monitor.batchLog), 0)
            self.assertIsNone(monitor.currentEpoch)
            self.assertIsNone(monitor.currentBatch)
            self.assertIsNone(monitor.bestEstimator)
            self.assertIsNone(monitor.bestEpoch)

        def check_assessment_empty(monitor):
            self.assertIsNone(monitor.assessmentModel)
            self.assertIsNone(monitor.asssessmentDataset)
            self.assertDictEqual(monitor.foldData, {})
            self.assertIsNone(monitor.predictions)
            self.assertDictEqual(monitor.estimators, {})

        def check_hyperparam_monitor(monitor):
            # calculate number of iterations from config
            n_iter = np.prod([len(v) for v in monitor.config["param_grid"].values()])
            self.assertGreater(n_iter, 0)
            self.assertEqual(len(monitor.assessments), n_iter)
            self.assertEqual(len(monitor.parameters), n_iter)
            self.assertEqual(monitor.scores.shape, (n_iter, 2))  # agg score + scores
            self.assertEqual(
                max(monitor.scores.aggregated_score),
                monitor.bestScore,
            )
            self.assertDictEqual(
                monitor.bestParameters,
                monitor.parameters[monitor.scores.aggregated_score.argmax()],
            )
            check_assessment_empty(monitor)
            check_fit_empty(monitor)

        def check_assessor_monitor(monitor, n_folds, len_y):
            self.assertEqual(
                monitor.predictions.shape,
                (len_y, 3 if n_folds > 1 else 2),  # labels + preds (+ fold)
            )
            self.assertEqual(len(monitor.foldData), n_folds)
            self.assertEqual(len(monitor.fits), n_folds)
            self.assertEqual(len(monitor.estimators), n_folds)
            check_fit_empty(monitor)

        def check_fit_monitor(monitor):
            self.assertGreater(len(monitor.fitLog), 0)
            self.assertGreater(len(monitor.batchLog), 0)
            self.assertTrue(isinstance(monitor.bestEstimator, monitor.fitModel.alg))
            self.assertIsNotNone(monitor.currentEpoch)
            self.assertIsNotNone(monitor.currentBatch)

        if monitor_type == "hyperparam":
            check_hyperparam_monitor(monitor)
        elif monitor_type == "crossval":
            check_assessor_monitor(monitor, 5, len(monitor.assessmentDataset.y))
        elif monitor_type == "test":
            check_assessor_monitor(monitor, 1, len(monitor.assessmentDataset.y_ind))
        elif monitor_type == "fit":
            if neural_net:
                check_fit_monitor(monitor)
            else:
                check_fit_empty(monitor)
        else:
            raise ValueError(f"Unknown monitor type {monitor_type}")

    def fileMonitorTest(
        self,
        monitor: FileMonitor,
        monitor_type: Literal["hyperparam", "crossval", "test", "fit"],
        neural_net: bool,
    ):
        """Test if the correct files are generated"""

        def check_fit_files(path):
            self.assertTrue(os.path.exists(f"{path}/fit_log.tsv"))
            self.assertTrue(os.path.exists(f"{path}/batch_log.tsv"))

        def check_assessment_files(path, monitor):
            output_path = f"{path}/{monitor.assessmentType}"
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(
                os.path.exists(
                    f"{output_path}/{monitor.assessmentType}_predictions.tsv"
                )
            )

            if monitor.saveFits and neural_net:
                for fold in monitor.foldData:
                    check_fit_files(f"{output_path}/fold_{fold}")

        def check_hyperparam_files(path, monitor):
            output_path = f"{path}/GridSearchOptimization"
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(
                os.path.exists(f"{output_path}/GridSearchOptimization_scores.tsv")
            )

            if monitor.saveAssessments:
                for assessment in monitor.assessments:
                    check_assessment_files(
                        f"{output_path}/iteration_{assessment}", monitor
                    )

        if monitor_type == "hyperparam":
            check_hyperparam_files(monitor.outDir, monitor)
        elif monitor_type in ["crossval", "test"]:
            check_assessment_files(monitor.outDir, monitor)
        elif monitor_type == "fit" and neural_net:
            check_fit_files(monitor.outDir)

    def listMonitorTest(
        self,
        monitor: ListMonitor,
        monitor_type: Literal["hyperparam", "crossval", "test", "fit"],
        neural_net: bool,
    ):
        self.baseMonitorTest(monitor.monitors[0], monitor_type, neural_net)
        self.fileMonitorTest(monitor.monitors[1], monitor_type, neural_net)

    def runMonitorTest(
        self, model, data, monitor_type, test_method, nerual_net, *args, **kwargs
    ):
        hyperparam_monitor = monitor_type(*args, **kwargs)
        crossval_monitor = deepcopy(hyperparam_monitor)
        test_monitor = deepcopy(hyperparam_monitor)
        fit_monitor = deepcopy(hyperparam_monitor)
        (
            hyperparam_monitor,
            crossval_monitor,
            test_monitor,
            fit_monitor,
        ) = self.trainModelWithMonitoring(
            model, data, hyperparam_monitor, crossval_monitor, test_monitor, fit_monitor
        )
        test_method(hyperparam_monitor, "hyperparam", nerual_net)
        test_method(crossval_monitor, "crossval", nerual_net)
        test_method(test_monitor, "test", nerual_net)
        test_method(fit_monitor, "fit", nerual_net)
