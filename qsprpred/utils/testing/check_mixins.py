import logging
import numbers
import os
from copy import deepcopy
from os.path import exists
from typing import Literal

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

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
            self.assertIsInstance(dataset.feature_standardizer, SKLearnStandardizer)
        else:
            self.assertIsNone(dataset.feature_standardizer)
        self.checkFeatures(dataset, expected_feature_count)
        # verify prep results are the same after reloading
        dataset.prepareDataset(
            feature_calculators=feature_calculators,
            split=split if split else None,
            feature_standardizer=feature_standardizer if feature_standardizer else None,
            feature_filters=[feature_filter] if feature_filter else None,
            data_filters=[data_filter] if data_filter else None,
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

    def fitTest(self, model: QSPRModel):
        """Test model fitting, optimization and evaluation.

        Args:
            model (QSPRModel): The model to test.
        """
        # perform bayes optimization
        score_func = "r2" if model.task.isRegression() else "roc_auc_ovr"
        search_space_bs = self.getParamGrid(model, "bayes")
        bayesoptimizer = OptunaOptimization(
            param_grid=search_space_bs,
            n_trials=1,
            model_assessor=CrossValAssessor(
                scoring=score_func, mode=EarlyStoppingMode.NOT_RECORDING
            ),
        )
        best_params = bayesoptimizer.optimize(model)
        model.setParams(best_params)
        model.save()
        model_new = model.__class__.fromFile(model.metaFile)
        for param in best_params:
            self.assertEqual(model_new.parameters[param], best_params[param])
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
        best_params = gridsearcher.optimize(model)
        model_new = model.__class__.fromFile(model.metaFile)
        for param in best_params:
            self.assertEqual(model_new.parameters[param], best_params[param])
        model.cleanFiles()
        # perform crossvalidation
        score_func = "r2" if model.task.isRegression() else "roc_auc_ovr"
        n_folds = 5
        scores = CrossValAssessor(
            mode=EarlyStoppingMode.RECORDING,
            scoring=score_func,
            split_multitask_scores=model.isMultiTask,
            split=KFold(
                n_splits=n_folds, shuffle=True, random_state=model.data.randomState
            ),
        )(model)
        if model.isMultiTask:
            self.assertEqual(scores.shape, (n_folds, len(model.targetProperties)))
        scores = TestSetAssessor(
            mode=EarlyStoppingMode.NOT_RECORDING,
            scoring=score_func,
            split_multitask_scores=model.isMultiTask,
        )(model)
        if model.isMultiTask:
            self.assertEqual(scores.shape, (len(model.targetProperties),))
        self.assertTrue(exists(f"{model.outDir}/{model.name}.ind.tsv"))
        self.assertTrue(exists(f"{model.outDir}/{model.name}.cv.tsv"))
        # train the model on all data
        path = model.fitAttached()
        self.assertTrue(exists(path))
        self.assertTrue(exists(model.metaFile))
        self.assertEqual(path, model.metaFile)

    def predictorTest(self,
            model: QSPRModel,
            dataset: QSPRDataset,
            comparison_model: QSPRModel | None = None,
            expect_equal_result=True):
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
        features = dataset.getFeatures(concat=True, ordered=True)
        expected_result = model.predict(features)

        # make predictions with the predictMols function
        smiles = dataset.getDF()[dataset.smilesCol].to_list()
        predictions = model.predictMols(smiles, use_probas=False)

        check_shape(predictions, model, len(smiles), use_probas=False)
        check_predictions(predictions, expected_result, True)

        # do the same for the predictProba function
        if model.task.isClassification():
            expected_result_proba = model.predictProba(features)
            predictions_proba = model.predictMols(smiles, use_probas=True)
            check_shape(predictions_proba, model, len(smiles), use_probas=True)
            check_predictions(predictions_proba, expected_result_proba, True)

        # check if the predictions are (not) the same as of the comparison model
        if comparison_model is not None:
            predictions_comparison = comparison_model.predictMols(smiles, use_probas=False)

            check_predictions(predictions, predictions_comparison, expect_equal_result)

            if model.task.isClassification():
                predictions_comparison_proba = comparison_model.predictMols(smiles, use_probas=True)
                check_predictions(predictions_proba, predictions_comparison_proba, expect_equal_result)

    def oldpredictorTest(self, model: QSPRModel, dataset: QSPRDataset, expected_result: list | np.ndarray, use_probas: bool=True, expect_equal_result=True):
        """Test model prediction.

        Args:
            model (QSPRModel): The model to make predictions with.
            dataset (QSPRDataset): The dataset to make predictions for.
            expected_result (list | np.ndarray): The expected result of the prediction.
            use_probas (bool): Whether to use `predictProbas` or `predict` method to
                make the predictions.
            expect_equal_result (bool): Whether the expected result should be equal or
                not equal to the prediction.
        """
        # define checks of the shape of the predictions
        def check_shape(predictions, model, num_smiles, use_probas):
            if model.task.isClassification() and use_probas:
                print(predictions)
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

        # make predictions
        smiles = dataset.getDF()[dataset.smilesCol].to_list()
        predictions = model.predictMols(smiles, use_probas=use_probas)

        # check the shape of the predictions
        check_shape(predictions, model, len(smiles), use_probas=use_probas)

        # make predictions with invalid smiles added to the list
        invalid_smiles = ["C1CCCCC1", "C1CCCCC"]
        predictions_with_invalids = model.predictMols(smiles + invalid_smiles, use_probas=use_probas)
        new_len = len(smiles) + len(invalid_smiles)

        # check the shape of the predictions
        check_shape(predictions_with_invalids, model, new_len, use_probas=use_probas)

        # check predictions are almost equal to expected result (rtol=1e-5)
        # or not equal to expected result for should_not_be_equal
        check_outcome = self.assertTrue if expect_equal_result else self.assertFalse
        if isinstance(expected_result, list):
            for i in range(len(expected_result)):
                check_outcome(np.allclose(predictions[i], expected_result[i]))
        else:
            check_outcome(np.allclose(predictions, expected_result))

        return predictions

    def oldoldpredictorTest(
        self,
        predictor: QSPRModel,
        expect_equal_result=True,
        expected_pred_use_probas=None,
        expected_pred_not_use_probas=None,
        **pred_kwargs,
    ):
        """Test a model as predictor.

        Args:
            predictor (QSPRModel):
                The model to test.
            expect_equal_result (bool):
                If pred values provided, specifies whether to check for equality or
                inequality.
            expected_pred_use_probas (float): Value to check with use_probas true.
            expected_pred_not_use_probas (int | float):
                Value to check with use_probas false. Ignored if expect_equal_result is
                false.
            **pred_kwargs:
                Extra keyword arguments to pass to the predictor's `predictMols` method.
        """

        # load molecules to predict
        df = pd.read_csv(
            f"{os.path.dirname(__file__)}/test_files/data/test_data.tsv", sep="\t"
        )

        # define checks of the shape of the predictions
        def check_shape(input_smiles):
            if predictor.targetProperties[0].task.isClassification() and use_probas:
                self.assertEqual(len(predictions), len(predictor.targetProperties))
                self.assertEqual(
                    predictions[0].shape,
                    (len(input_smiles), predictor.targetProperties[0].nClasses),
                )
            else:
                self.assertEqual(
                    predictions.shape,
                    (len(input_smiles), len(predictor.targetProperties)),
                )

        # check the predictions for different settings of use_probas
        pred = []
        for use_probas in [True, False]:
            # make predictions
            predictions = predictor.predictMols(
                df.SMILES.to_list(), use_probas=use_probas, **pred_kwargs
            )
            # check the shape of the predictions
            check_shape(df.SMILES.to_list())
            # check the type of the predictions
            if isinstance(predictions, list):
                for prediction in predictions:
                    self.assertIsInstance(prediction, np.ndarray)
            else:
                self.assertIsInstance(predictions, np.ndarray)

            # check the first predicted value
            singleoutput = (
                predictions[0][0, 0]
                if isinstance(predictions, list)
                else predictions[0, 0]
            )
            # check the type of the first predicted value depending on the task
            if (
                predictor.targetProperties[0].task == TargetTasks.REGRESSION
                or use_probas
            ):
                self.assertIsInstance(singleoutput, numbers.Real)
            elif predictor.targetProperties[
                0
            ].task == TargetTasks.MULTICLASS or isinstance(
                predictor.estimator, XGBClassifier
            ):
                self.assertIsInstance(singleoutput, numbers.Integral)
            elif predictor.targetProperties[0].task == TargetTasks.SINGLECLASS:
                self.assertIn(singleoutput, [1, 0])
            else:
                return AssertionError(f"Unknown task: {predictor.task}")
            pred.append(singleoutput)
            # test with invalid smiles
            invalid_smiles = ["C1CCCCC1", "C1CCCCC"]
            predictions = predictor.predictMols(
                invalid_smiles, use_probas=use_probas, **pred_kwargs
            )
            check_shape(invalid_smiles)
            # check that the first prediction is None
            singleoutput = (
                predictions[0][0, 0]
                if isinstance(predictions, list)
                else predictions[0, 0]
            )
            self.assertEqual(
                predictions[0][1, 0]
                if isinstance(predictions, list)
                else predictions[1, 0],
                None,
            )
            # check the type of the first predicted value depending on the task
            if (
                predictor.targetProperties[0].task == TargetTasks.SINGLECLASS
                and not isinstance(predictor.estimator, XGBClassifier)
                and not use_probas
            ):
                self.assertIn(singleoutput, [0, 1])
            else:
                self.assertIsInstance(singleoutput, numbers.Number)

        # check that the predictions are the same for use_probas=True and False as
        # expected
        if expect_equal_result:
            if expected_pred_use_probas is not None:
                self.assertAlmostEqual(pred[0], expected_pred_use_probas, places=8)
            if expected_pred_not_use_probas is not None:
                self.assertAlmostEqual(pred[1], expected_pred_not_use_probas, places=8)
        elif expected_pred_use_probas is not None:
            # Skip not_use_probas test:
            # For regression this is identical to use_probas result
            # For classification there is no guarantee the classification result
            # actually differs, unlike probas which will usually have different results
            self.assertNotAlmostEqual(pred[0], expected_pred_use_probas, places=8)

        return pred[0], pred[1]

    # NOTE: below code is taken from CorrelationPlot without the plotting code
    # Would be nice if the summary code without plot was available regardless
    def createCorrelationSummary(self, model):
        cv_path = f"{model.outPrefix}.cv.tsv"
        ind_path = f"{model.outPrefix}.ind.tsv"

        cate = [cv_path, ind_path]
        cate_names = ["cv", "ind"]
        property_name = model.targetProperties[0].name
        summary = {"ModelName": [], "R2": [], "RMSE": [], "Set": []}
        for j, _ in enumerate(["Cross Validation", "Independent Test"]):
            df = pd.read_table(cate[j])
            coef = metrics.r2_score(
                df[f"{property_name}_Label"], df[f"{property_name}_Prediction"]
            )
            rmse = metrics.mean_squared_error(
                df[f"{property_name}_Label"],
                df[f"{property_name}_Prediction"],
                squared=False,
            )
            summary["R2"].append(coef)
            summary["RMSE"].append(rmse)
            summary["Set"].append(cate_names[j])
            summary["ModelName"].append(model.name)

        return summary

    # NOTE: below code is taken from MetricsPlot without the plotting code
    # Would be nice if the summary code without plot was available regardless
    def createMetricsSummary(self, model):
        decision_threshold: float = 0.5
        metrics = [
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            accuracy_score,
        ]
        summary = {"Metric": [], "Model": [], "TestSet": [], "Value": []}
        property_name = model.targetProperties[0].name

        cv_path = f"{model.outPrefix}.cv.tsv"
        ind_path = f"{model.outPrefix}.ind.tsv"

        df = pd.read_table(cv_path)

        # cross-validation
        for fold in sorted(df.Fold.unique()):
            y_pred = df[f"{property_name}_ProbabilityClass_1"][df.Fold == fold]
            y_pred_values = [1 if x > decision_threshold else 0 for x in y_pred]
            y_true = df[f"{property_name}_Label"][df.Fold == fold]
            for metric in metrics:
                val = metric(y_true, y_pred_values)
                summary["Metric"].append(metric.__name__)
                summary["Model"].append(model.name)
                summary["TestSet"].append(f"CV{fold + 1}")
                summary["Value"].append(val)

        # independent test set
        df = pd.read_table(ind_path)
        y_pred = df[f"{property_name}_ProbabilityClass_1"]
        th = 0.5
        y_pred_values = [1 if x > th else 0 for x in y_pred]
        y_true = df[f"{property_name}_Label"]
        for metric in metrics:
            val = metric(y_true, y_pred_values)
            summary["Metric"].append(metric.__name__)
            summary["Model"].append(model.name)
            summary["TestSet"].append("IND")
            summary["Value"].append(val)

        return summary


class MonitorsCheckMixIn(ModelDataSetsPathMixIn, ModelCheckMixIn):
    def trainModelWithMonitoring(
        self,
        model: QSPRModel,
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
        score_func = "r2" if model.task.isRegression() else "roc_auc_ovr"
        search_space_gs = self.getParamGrid(model, "grid")
        gridsearcher = GridSearchOptimization(
            param_grid=search_space_gs,
            model_assessor=CrossValAssessor(
                scoring=score_func,
                mode=EarlyStoppingMode.NOT_RECORDING,
            ),
            monitor=hyperparam_monitor,
        )
        best_params = gridsearcher.optimize(model)
        model.setParams(best_params)
        model.save()
        # perform crossvalidation
        CrossValAssessor(
            mode=EarlyStoppingMode.RECORDING,
            scoring=score_func,
            monitor=crossval_monitor,
        )(model)
        TestSetAssessor(
            mode=EarlyStoppingMode.NOT_RECORDING,
            scoring=score_func,
            monitor=test_monitor,
        )(model)
        # train the model on all data
        model.fitAttached(monitor=fit_monitor)
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
        self, model, monitor_type, test_method, nerual_net, *args, **kwargs
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
            model, hyperparam_monitor, crossval_monitor, test_monitor, fit_monitor
        )
        test_method(hyperparam_monitor, "hyperparam", nerual_net)
        test_method(crossval_monitor, "crossval", nerual_net)
        test_method(test_monitor, "test", nerual_net)
        test_method(fit_monitor, "fit", nerual_net)
