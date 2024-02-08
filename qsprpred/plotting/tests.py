"""Tests for plotting module."""

import os
from typing import Type

import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from parameterized import parameterized
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from qsprpred.models.assessment.methods import CrossValAssessor, TestSetAssessor
from ..data.processing.feature_filters import LowVarianceFilter
from ..data.tables.qspr import QSPRDataset
from ..models.scikit_learn import SklearnModel
from ..plotting.classification import ConfusionMatrixPlot, MetricsPlot, ROCPlot
from ..plotting.regression import CorrelationPlot, WilliamsPlot
from ..tasks import TargetTasks
from ..utils.testing.base import QSPRTestCase
from ..utils.testing.path_mixins import ModelDataSetsPathMixIn


class PlottingTest(ModelDataSetsPathMixIn, QSPRTestCase):
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def getModel(self, name: str, alg: Type = RandomForestClassifier) -> SklearnModel:
        """Get a model for testing.

        Args:
            dataset (QSPRDataset):
                Dataset to use for model.
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
            target_props=[{"name": "CL", "task": TargetTasks.SINGLECLASS, "th": [6.5]}],
            preparation_settings=self.getDefaultPrep(),
        )
        model = self.getModel("test_roc_plot_single_model")
        score_func = "roc_auc_ovr"
        CrossValAssessor(scoring=score_func)(model, dataset)
        TestSetAssessor(scoring=score_func)(model, dataset)
        model.save()
        # make plots
        plt = ROCPlot([model])
        # cross validation plot
        ax = plt.make(validation="cv")[0]
        self.assertIsInstance(ax, Figure)
        self.assertTrue(os.path.exists(f"{model.outPrefix}.cv.png"))
        # independent test set plot
        ax = plt.make(validation="ind")[0]
        self.assertIsInstance(ax, Figure)
        self.assertTrue(os.path.exists(f"{model.outPrefix}.ind.png"))


class MetricsPlotTest(PlottingTest):
    """Test metrics plotting class."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    @parameterized.expand(
        [
            (task, task, th)
            for task, th in (
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
                    "task": TargetTasks.SINGLECLASS
                    if task == "binary"
                    else TargetTasks.MULTICLASS,
                    "th": th,
                }
            ],
            preparation_settings=self.getDefaultPrep(),
        )
        model = self.getModel(f"test_metrics_plot_single_{task}_model")
        score_func = "roc_auc_ovr"
        CrossValAssessor(scoring=score_func)(model, dataset)
        TestSetAssessor(scoring=score_func)(model, dataset)
        model.save()
        # generate metrics plot and associated files
        plt = MetricsPlot([model])
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
            "test_corr_plot_single_data", preparation_settings=self.getDefaultPrep()
        )
        model = self.getModel("test_corr_plot_single_model", alg=RandomForestRegressor)
        score_func = "r2"
        CrossValAssessor(scoring=score_func)(model, dataset)
        TestSetAssessor(scoring=score_func)(model, dataset)
        model.save()
        # generate metrics plot and associated files
        plt = CorrelationPlot([model])
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
            "test_williams_plot_single_data", preparation_settings=self.getDefaultPrep()
        )
        # filter features to below the number of samples in the test set
        # to avoid error in WilliamsPlot
        dataset.filterFeatures([LowVarianceFilter(0.23)])
        model = self.getModel(
            "test_williams_plot_single_model", alg=RandomForestRegressor
        )
        score_func = "r2"
        CrossValAssessor(scoring=score_func)(model, dataset)
        TestSetAssessor(scoring=score_func)(model, dataset)
        model.save()
        # generate metrics plot and associated files
        plt = WilliamsPlot([model], [dataset])
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
            (task, task, th)
            for task, th in (
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
                    "task": TargetTasks.SINGLECLASS
                    if task == "binary"
                    else TargetTasks.MULTICLASS,
                    "th": th,
                }
            ],
            preparation_settings=self.getDefaultPrep(),
        )
        model = self.getModel(f"test_cm_plot_single_{task}_model")
        score_func = "roc_auc_ovr"
        CrossValAssessor(scoring=score_func)(model, dataset)
        TestSetAssessor(scoring=score_func)(model, dataset)
        model.save()
        # make plots
        plt = ConfusionMatrixPlot([model])
        axes, cm_dict = plt.make()
        # assert all figures are sns.FacetGrid
        for ax in axes:
            self.assertIsInstance(ax, Figure)
        self.assertIsInstance(cm_dict, dict)
        self.assertTrue(
            os.path.exists(f"{model.outPrefix}_CL_0.0_confusion_matrix.png")
        )
        self.assertTrue(
            os.path.exists(f"{model.outPrefix}_CL_1.0_confusion_matrix.png")
        )
        self.assertTrue(
            os.path.exists(f"{model.outPrefix}_CL_2.0_confusion_matrix.png")
        )
        self.assertTrue(
            os.path.exists(f"{model.outPrefix}_CL_3.0_confusion_matrix.png")
        )
        self.assertTrue(
            os.path.exists(f"{model.outPrefix}_CL_4.0_confusion_matrix.png")
        )
        self.assertTrue(
            os.path.exists(
                f"{model.outPrefix}_CL_Independent Test_confusion_matrix.png"
            )
        )
