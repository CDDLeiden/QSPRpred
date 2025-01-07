"""Tests for plotting module."""

import os
from typing import Type

import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from parameterized import parameterized
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from qsprpred.models.assessment.methods import Assessor

from ..data.processing.feature_filters import LowVarianceFilter
from ..models.scikit_learn import SklearnModel
from ..plotting.classification import ConfusionMatrixPlot, MetricsPlot, ROCPlot
from ..plotting.regression import CorrelationPlot, WilliamsPlot
from ..tasks import TargetTasks
from ..utils.testing.base import QSPRTestCase
from ..utils.testing.path_mixins import ModelDataSetsPathMixIn
from sklearn.model_selection import KFold
from ..data.sampling.splits import RandomSplit


class PlottingTest(ModelDataSetsPathMixIn, QSPRTestCase):
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def getModel(self, name: str, alg: Type = RandomForestClassifier) -> SklearnModel:
        """Get a model for testing.

        Args:
            name (str):
                Name of model.
            alg (Type, optional):
                Algorithm to use for model. Defaults to `RandomForestClassifier`.

        Returns:
            SklearnModel:
                The new model.

        """
        return SklearnModel(
            name=name,
            base_dir=self.generatedModelsPath,
            alg=alg,
        )


class ROCPlotTest(PlottingTest):
    """Test ROC curve plotting class."""
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def testPlotSingle(self):
        """Test plotting ROC curve for single task."""
        dataset = self.createLargeTestDataSet(
            "test_roc_plot_single_data",
            target_props=[{
                "name": "CL",
                "task": TargetTasks.SINGLECLASS,
                "th": [6.5]
            }],
        )
        model = self.getModel("test_roc_plot_single_model")
        score_func = "roc_auc_ovr"
        Assessor(
            "crossval",
            split = KFold(n_splits=5, shuffle=True, random_state=model.randomState),
            scoring = score_func
        )(model, dataset, pipeline=self.getDefaultPrep())
        Assessor(
            "test",
            split=RandomSplit(test_fraction=0.2),
            scoring=score_func)(model, dataset, pipeline=self.getDefaultPrep())
        model.save()
        # make plots
        plt = ROCPlot([model], ["crossval"])
        # cross validation plot
        ax = plt.make()[0]
        self.assertIsInstance(ax, Figure)
        self.assertTrue(os.path.exists(f"{model.outPrefix}_crossval_ROC.png"))
        # independent test set plot
        plt = ROCPlot([model], ["test"])
        ax = plt.make()[0]
        self.assertIsInstance(ax, Figure)
        self.assertTrue(os.path.exists(f"{model.outPrefix}_test_ROC.png"))


class MetricsPlotTest(PlottingTest):
    """Test metrics plotting class."""
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    @parameterized.expand(
        [
            (task, task, th) for task, th in (
                ("binary", [6.5]),
                ("multi_class", [0, 2, 10, 1100]),
            )
        ]
    )
    def testPlotSingle(self, _, task, th):
        """Test plotting metrics for single task single class and multi-class."""
        dataset = self.createLargeTestDataSet(
            f"test_metrics_plot_single_{task}_data",
            target_props=[
                {
                    "name": "CL",
                    "task":
                        (
                            TargetTasks.SINGLECLASS
                            if task == "binary" else TargetTasks.MULTICLASS
                        ),
                    "th": th,
                }
            ],
        )
        model = self.getModel(f"test_metrics_plot_single_{task}_model")
        score_func = "roc_auc_ovr"
        Assessor(
            "crossval",
            split = KFold(n_splits=5, shuffle=True, random_state=model.randomState),
            scoring = score_func
        )(model, dataset, pipeline=self.getDefaultPrep())
        Assessor(
            "test",
            split=RandomSplit(test_fraction=0.2),
            scoring=score_func)(model, dataset, pipeline=self.getDefaultPrep())
        model.save()
        # generate metrics plot and associated files
        plt = MetricsPlot([model], ["crossval", "test"])
        figures, summary = plt.make()
        for g in figures:
            self.assertIsInstance(g, sns.FacetGrid)
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertTrue(os.path.exists(f"{model.outPrefix}_precision.png"))


class CorrPlotTest(PlottingTest):
    """Test correlation plotting class."""
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def testPlotSingle(self):
        """Test plotting correlation for single task."""
        dataset = self.createLargeTestDataSet(
            "test_corr_plot_single_data"
        )
        model = self.getModel("test_corr_plot_single_model", alg=RandomForestRegressor)
        score_func = "r2"
        Assessor(
            "crossval",
            split = KFold(n_splits=5, shuffle=True, random_state=model.randomState),
            scoring = score_func
        )(model, dataset, pipeline=self.getDefaultPrep())
        Assessor(
            "test",
            split=RandomSplit(test_fraction=0.2),
            scoring=score_func)(model, dataset, pipeline=self.getDefaultPrep())
        model.save()
        # generate metrics plot and associated files
        plt = CorrelationPlot([model], ["crossval", "test"])
        g, summary = plt.make("CL")
        self.assertIsInstance(summary, pd.DataFrame)
        # assert g is sns.FacetGrid
        self.assertIsInstance(g, sns.FacetGrid)
        self.assertTrue(os.path.exists(f"{model.outPrefix}_correlation.png"))


class WilliamsPlotTest(PlottingTest):
    """Test plotting Williams plot for single task."""
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def testPlotSingle(self):
        """Test plotting Williams plot for single task."""
        dataset = self.createLargeTestDataSet(
            "test_williams_plot_single_data",
        )
        pipeline = self.getDefaultPrep()
        pipeline.steps.update({"low_variance_filter": LowVarianceFilter(0.23)})
        # filter features to below the number of samples in the test set
        # to avoid error in WilliamsPlot
        model = self.getModel(
            "test_williams_plot_single_model", alg=RandomForestRegressor
        )
        score_func = "r2"
        Assessor(
            "crossval",
            split = KFold(n_splits=5, shuffle=True, random_state=model.randomState),
            scoring = score_func
        )(model, dataset, pipeline=self.getDefaultPrep())
        Assessor(
            "test",
            split=RandomSplit(test_fraction=0.2),
            scoring=score_func)(model, dataset, pipeline=self.getDefaultPrep())
        model.fitDataset(dataset, pipeline)
        model.save()
        # generate metrics plot and associated files
        plt = WilliamsPlot([model], [dataset], ["crossval", "test"])
        g, leverages, hstar = plt.make()
        self.assertIsInstance(leverages, pd.DataFrame)
        self.assertIsInstance(hstar, dict)
        # assert g is sns.FacetGrid
        self.assertIsInstance(g, sns.FacetGrid)
        self.assertTrue(os.path.exists(f"{model.outPrefix}_williamsplot.png"))


class ConfusionMatrixPlotTest(PlottingTest):
    """Test confusion matrix plotting class."""
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    @parameterized.expand(
        [
            (task, task, th) for task, th in (
                ("binary", [6.5]),
                ("multi_class", [0, 2, 10, 1100]),
            )
        ]
    )
    def testPlotSingle(self, _, task, th):
        """Test plotting confusion matrix for single task."""
        dataset = self.createLargeTestDataSet(
            f"test_cm_plot_single_{task}_data",
            target_props=[
                {
                    "name": "CL",
                    "task":
                        (
                            TargetTasks.SINGLECLASS
                            if task == "binary" else TargetTasks.MULTICLASS
                        ),
                    "th": th,
                }
            ],
        )
        model = self.getModel(f"test_cm_plot_single_{task}_model")
        score_func = "roc_auc_ovr"
        Assessor(
            "crossval",
            split = KFold(n_splits=5, shuffle=True, random_state=model.randomState),
            scoring = score_func
        )(model, dataset, pipeline=self.getDefaultPrep())
        Assessor(
            "test",
            split=RandomSplit(test_fraction=0.2),
            scoring=score_func)(model, dataset, pipeline=self.getDefaultPrep())
        model.save()
        # make plots
        plt = ConfusionMatrixPlot([model], ["crossval"])
        axes, cm_dict = plt.make()
        # assert all figures are sns.FacetGrid
        for ax in axes:
            self.assertIsInstance(ax, Figure)
        self.assertIsInstance(cm_dict, dict)
        self.assertTrue(
            os.path.exists(f"{model.outPrefix}_CL_crossval_0_confusion_matrix.png")
        )
        self.assertTrue(
            os.path.exists(f"{model.outPrefix}_CL_crossval_1_confusion_matrix.png")
        )
        self.assertTrue(
            os.path.exists(f"{model.outPrefix}_CL_crossval_2_confusion_matrix.png")
        )
        self.assertTrue(
            os.path.exists(f"{model.outPrefix}_CL_crossval_3_confusion_matrix.png")
        )
        self.assertTrue(
            os.path.exists(f"{model.outPrefix}_CL_crossval_4_confusion_matrix.png")
        )
        plt = ConfusionMatrixPlot([model], ["test"])
        axes, cm_dict = plt.make()
        for ax in axes:
            self.assertIsInstance(ax, Figure)
        self.assertIsInstance(cm_dict, dict)
        self.assertTrue(
            os.path.
            exists(f"{model.outPrefix}_CL_test_0_confusion_matrix.png")
        )
