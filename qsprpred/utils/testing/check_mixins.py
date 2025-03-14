import logging
import os
from copy import deepcopy
from os.path import exists
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ... import TargetTasks
from ...data.descriptors.sets import DescriptorSet
from ...data.processing.feature_filters import FeatureFilter
from ...data.processing.pipeline import DatasetPipeline, DummyStep, SklearnStep
from ...data.tables.interfaces.qspr_data_set import QSPRDataSet
from ...models import (
    AssessorMonitor,
    BaseMonitor,
    Assessor,
    EarlyStoppingMode,
    FileMonitor,
    FitMonitor,
    GridSearchOptimization,
    HyperparameterOptimization,
    HyperparameterOptimizationMonitor,
    OptunaOptimization,
    QSPRModel,
    SklearnMetrics,
    Assessor,
)
from ...models.monitors import ListMonitor
from ...tasks import TargetProperty
from .path_mixins import ModelDataSetsPathMixIn
from ...data.sampling.splits import DataSplit, RandomSplit

class PipelineStepCheckMixIn:
    """Mixin class for common pipeline step checks."""
    def checkStep(self, step, dataset):
        """Check if the step is a valid pipeline step."""
        pass

    def checkStepInPipeline(self, step, dataset):
        """Check if the step is in the pipeline."""
        pass

class DescriptorCheckMixIn:
    """Mixin class for common descriptor checks."""
    def checkFeatures(self, X_train, y_train, X_test = None, y_test = None):
        """Check if features matrices are the correct type and shape and if the indices
        are consistent between features and targets. Also check if there is no overlap
        between the train and test indices if both are provided.
        """
        self.assertTrue(isinstance(X_train, pd.DataFrame))
        self.assertTrue(isinstance(y_train, pd.DataFrame))
        self.assertTrue(X_train.shape[0] == y_train.shape[0])
        self.assertTrue(X_train.index.equals(y_train.index))
        
        if X_test is not None and y_test is not None:
            self.assertTrue(isinstance(X_test, pd.DataFrame))
            self.assertTrue(isinstance(y_test, pd.DataFrame))
            self.assertTrue(X_test.shape[0] == y_test.shape[0])
            self.assertTrue(X_test.index.equals(y_test.index))
            self.assertTrue(X_train.shape[1] == X_test.shape[1])
            self.assertTrue(y_train.shape[1] == y_test.shape[1])
            self.assertTrue(X_train.index.intersection(X_test.index).empty)
    
    def checkDescriptors(
        self, dataset: QSPRDataSet, target_props: list[dict | TargetProperty]
    ):
        """Check if information about descriptors is consistent in the data set. Checks
        if calculators are consistent with the descriptors contained in the data set.
        This is tested also before and after serialization.

        Args:
            dataset (QSPRDataSet): The data set to check.
            target_props (List of dicts or TargetProperty): list of target properties

        Raises:
            AssertionError: If the consistency check fails.
        """
        # check if the descriptors are consistent with getDescriptors method
        expected_length = 0
        for calc in dataset.descriptorSets:
            expected_length += len(calc.descriptors)
        self.assertEqual(len(dataset.getDescriptors()), expected_length)

        dataset.save()
        ds_loaded = dataset.__class__.fromFile(dataset.metaFile)
        
        # check randomState, targetProperties and descriptorSets are loaded correctly
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
        self.assertEqual(len(ds_loaded.getDescriptors()), expected_length)

class DataPrepCheckMixIn(DescriptorCheckMixIn):
    """Mixin for testing data preparation."""
    def checkPrep(
        self,
        dataset: QSPRDataSet,
        pipeline: DatasetPipeline,
        split: DataSplit | None = None,
    ):
        """Check if the data preparation is consistent before and after reloading"""
        def checkIdenticalFeatures(features1, features2):
            """check that two sets of features and targets are identical
            
            Args:
                features1 (tuple(pd.Dataframe)): (X_train, y_train, X_test, y_test)
                features2 (tuple(pd.Dataframe)): (X_train, y_train, X_test, y_test)
            """
            for f1, f2 in zip(features1, features2):
                if f1 is not None and f2 is not None:
                    self.assertTrue(f1.index.equals(f2.index))
                    self.assertTrue(f1.columns.equals(f2.columns))
                    self.assertTrue(f1.equals(f2))
        # check if the features are the correct type and shape
        feature_list = []
        for features in pipeline.apply(dataset, split):
            self.checkFeatures(*features)
            feature_list.append(features)
        
        # check if the features are the same after reloading the dataset
        dataset.save()
        dataset_reload = dataset.__class__.fromFile(dataset.metaFile)
        for i, features in enumerate(pipeline.apply(dataset_reload, split, fit=False)):
            self.checkFeatures(*features)
            checkIdenticalFeatures(features, feature_list[i])
            
        # check if the features are the same after reloading the pipeline
        pipeline.toFile(f"{dataset.path}_pipeline.json")
        pipeline_reload = DatasetPipeline.fromFile(f"{dataset.path}_pipeline.json")
        for i, features in enumerate(pipeline_reload.apply(dataset, split, fit=False)):
            self.checkFeatures(*features)
            checkIdenticalFeatures(features, feature_list[i])

    def checkSplit(self, dataset: QSPRDataSet, name: str):
        """Check if the split has the data it should have after splitting."""
        self.assertTrue(isinstance(dataset.getSplit(name), DataSplit)) 
        
        for X_train, y_train, X_test, y_test in dataset.iterSplit(name, as_type="pandas"):
            self.checkFeatures(X_train, y_train, X_test, y_test)

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
        self,
        model: QSPRModel,
        ds: QSPRDataSet,
        pipeline: DatasetPipeline,
        optimizer: HyperparameterOptimization
    ):
        model_path, est_path = model.save(save_estimator=True)
        # get last modified time stamp of the model file
        model_last_modified = os.path.getmtime(est_path)
        best_params = optimizer.optimize(model, ds, pipeline)
        for param in best_params:
            self.assertEqual(best_params[param], model.parameters[param])
        new_time_modified = os.path.getmtime(est_path)
        self.assertTrue(model_last_modified < new_time_modified)
        optimizer.optimize(model, ds, pipeline, refit_optimal=True)
        model_last_modified = new_time_modified
        new_time_modified = os.path.getmtime(est_path)
        self.assertTrue(model_last_modified < new_time_modified)
        model_new = model.__class__.fromFile(model.metaFile)
        for param in model.parameters:
            self.assertEqual(model_new.parameters[param], model.parameters[param])

    def fitTest(self, model: QSPRModel, ds: QSPRDataSet, pipeline: DatasetPipeline):
        """Test model fitting, optimization and evaluation.

        Args:
            model (QSPRModel): The model to test.
            ds (QSPRDataSet): The dataset to use for testing.
            pipeline (DatasetPipeline): The pipeline to use for testing.
        """
        # perform bayes optimization
        model.initFromData(ds, pipeline)
        score_func = "r2" if model.task.isRegression() else "roc_auc_ovr"
        search_space_bs = self.getParamGrid(model, "bayes")
        bayesoptimizer = OptunaOptimization(
            param_grid=search_space_bs,
            n_trials=1,
            model_assessor=Assessor(
                name="optuna_crossval",
                split=KFold(n_splits=5, shuffle=True, random_state=model.randomState),
                scoring=score_func,
                mode=EarlyStoppingMode.NOT_RECORDING
            ),
        )
        self.checkOptimization(model, ds, pipeline, bayesoptimizer)
        model.cleanFiles()
        # perform grid search
        search_space_gs = self.getParamGrid(model, "grid")
        if model.task.isClassification():
            score_func = SklearnMetrics("accuracy")
        gridsearcher = GridSearchOptimization(
            param_grid=search_space_gs,
            score_aggregation=np.median,
            model_assessor=Assessor(
                name="grid_test",
                split=RandomSplit(test_fraction=0.2),
                scoring=score_func,
                use_proba=False,
                mode=EarlyStoppingMode.NOT_RECORDING,
            ),
        )
        self.checkOptimization(model, ds, pipeline, gridsearcher)
        model.cleanFiles()
        # perform crossvalidation
        score_func = "r2" if model.task.isRegression() else "roc_auc_ovr"
        n_folds = 5
        cross_val = Assessor(
            name="crossval",
            scoring=score_func,
            split=KFold(n_splits=n_folds, shuffle=True, random_state=model.randomState),
            mode=EarlyStoppingMode.RECORDING,
            split_multitask_scores=model.isMultiTask,
        )
        scores = cross_val(model, ds, pipeline)
        if model.isMultiTask:
            self.assertEqual(scores.shape, (n_folds, len(model.targetProperties)))
        test_set = Assessor(
            name="test",
            scoring=score_func,
            split=RandomSplit(test_fraction=0.2),
            mode=EarlyStoppingMode.NOT_RECORDING,
            split_multitask_scores=model.isMultiTask,
        )
        scores = test_set(model, ds, pipeline)
        if model.isMultiTask:
            self.assertEqual(scores.shape, (1, len(model.targetProperties)))
        self.assertTrue(exists(f"{model.outDir}/{model.name}_crossval.tsv"))
        self.assertTrue(exists(f"{model.outDir}/{model.name}_test.tsv"))
        # train the model on all data
        path = model.fitDataset(ds, pipeline)
        self.assertTrue(exists(path))
        self.assertTrue(exists(model.metaFile))
        self.assertEqual(path, model.metaFile)

    def predictorTest(
        self,
        model: QSPRModel,
        dataset: QSPRDataSet,
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
            dataset (QSPRDataSet): The dataset to make predictions for.
            comparison_model (QSPRModel): another model to compare the predictions with.
            expect_equal_result (bool): Whether the expected result should be equal or
                not equal to the predictions of the comparison model.
            **pred_kwargs:
                Extra keyword arguments to pass to the predictor's `predictMols` method.
        """
        def reorder_predictions(predictions: np.ndarray, order: pd.Index, dataset: QSPRDataSet):
            """Reorder the predictions according to the order of the dataset."""
            if isinstance(predictions, list):
                predictions = [
                    pd.DataFrame(pred, index=order)
                    .loc[dataset.getDF().index.intersection(order)]
                    .values for pred in predictions
                ]
            else:
                predictions = (
                    pd.DataFrame(predictions, index=order)
                    .loc[dataset.getDF().index.intersection(order)]
                    .values
                )
            return predictions

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
        def check_predictions(preds, expected, expect_equal):
            # check if predictions are almost equal to expected result (rtol=1e-5)
            check_outcome = self.assertTrue if expect_equal else self.assertFalse
            if isinstance(expected, list):
                for i in range(len(expected)):
                    check_outcome(np.allclose(preds[i], expected[i]))
            else:
                check_outcome(np.allclose(preds, expected))

        # Check if the predictMols function gives the same result as the
        # predict/predictProba function
        # get the expected result from the basic predict function
        X, _ = next(model.pipeline.apply(dataset, fit=False))
        expected_result = model.predict(X)
        expected_result = reorder_predictions(expected_result, X.index, dataset)
        # make predictions with the predictMols function and check with previous result
        smiles = list(dataset.smiles)
        num_smiles = len(smiles)
        predictions = model.predictMols(smiles, use_probas=False, **pred_kwargs)
        check_shape(predictions, model, num_smiles, use_probas=False)
        check_predictions(predictions, expected_result, True)
        # do the same for the predictProba function
        predictions_proba = None
        if model.task.isClassification():
            expected_result_proba = model.predictProba(X)
            expected_result_proba = reorder_predictions(
                expected_result_proba, X.index, dataset
            )
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
        ds: QSPRDataSet,
        pipeline: DatasetPipeline,
        hyperparam_monitor: HyperparameterOptimizationMonitor,
        crossval_monitor: AssessorMonitor,
        test_monitor: AssessorMonitor,
        fit_monitor: FitMonitor,
    ) -> Tuple[
        HyperparameterOptimizationMonitor,
        AssessorMonitor,
        AssessorMonitor,
        FitMonitor,
    ]:
        score_func = (
            "r2" if ds.targetProperties[0].task.isRegression() else "roc_auc_ovr"
        )
        search_space_gs = self.getParamGrid(model, "grid")
        gridsearcher = GridSearchOptimization(
            param_grid=search_space_gs,
            model_assessor=Assessor(
                name="grid_test",
                split=RandomSplit(test_fraction=0.2),
                scoring=score_func,
                mode=EarlyStoppingMode.NOT_RECORDING,
            ),
            monitor=hyperparam_monitor,
        )
        best_params = gridsearcher.optimize(model, ds, pipeline)
        model.setParams(best_params)
        model.save()
        # perform crossvalidation
        Assessor(
            name="crossval",
            split=KFold(n_splits=5, shuffle=True, random_state=model.randomState),
            mode=EarlyStoppingMode.RECORDING,
            scoring=score_func,
            monitor=crossval_monitor,
        )(model, ds, pipeline)
        Assessor(
            name="test",
            split=RandomSplit(test_fraction=0.2),
            mode=EarlyStoppingMode.NOT_RECORDING,
            scoring=score_func,
            monitor=test_monitor,
        )(model, ds, pipeline)
        # train the model on all data
        model.fitDataset(ds, monitor=fit_monitor, pipeline=pipeline)
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
                (len_y, 5),  # labels + dataset labels + preds + fold + set
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
            # length should be the number of folds times the total length of the dataset
            # as both the training and test set are stored for each fold
            check_assessor_monitor(monitor, 5, len(monitor.assessmentDataset.getTargets())*5)
        elif monitor_type == "test":
            check_assessor_monitor(monitor, 1, len(monitor.assessmentDataset.getTargets()))
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
            assessment_name = monitor.assessmentName if hasattr(monitor, "assessmentName") else monitor["assessmentName"]
            output_path = f"{path}/{assessment_name}"
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(
                os.path.
                exists(f"{output_path}/{assessment_name}_settings.json")
            )
            self.assertTrue(
                os.path.
                exists(f"{output_path}/{assessment_name}_predictions.tsv")
            )

            save_fits = monitor.saveFits if hasattr(monitor, "saveFits") else monitor["saveFits"]
            if save_fits and neural_net:
                fold_data = monitor.foldData if hasattr(monitor, "foldData") else monitor["foldData"]
                for fold in fold_data:
                    check_fit_files(f"{output_path}/fold_{fold}")

        def check_hyperparam_files(path, monitor):
            output_path = f"{path}/GridSearchOptimization"
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(
                os.path.exists(f"{output_path}/GridSearchOptimization_scores.tsv")
            )

            if monitor.saveAssessments:
                for idx, assessment in monitor.assessments.items():
                    assessment["saveFits"] = monitor.saveFits
                    check_assessment_files(
                        f"{output_path}/iteration_{idx}", assessment
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
        self, model, data, pipeline, monitor_type, test_method, neural_net, *args, **kwargs
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
            model, data, pipeline, hyperparam_monitor, crossval_monitor, test_monitor, fit_monitor
        )
        test_method(hyperparam_monitor, "hyperparam", neural_net)
        test_method(crossval_monitor, "crossval", neural_net)
        test_method(test_monitor, "test", neural_net)
        test_method(fit_monitor, "fit", neural_net)
