"""Tests for plotting module."""

import os
from typing import Type
from unittest import TestCase

import pandas as pd
from matplotlib.axes import SubplotBase
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score

from ..data.data import QSPRDataset
from ..models.assessment_methods import CrossValAssessor, TestSetAssessor
from ..models.sklearn import SklearnModel
from ..models.tasks import TargetTasks
from ..models.tests import ModelDataSetsMixIn
from ..models.metrics import SklearnMetrics
from ..plotting.classification import MetricsPlot, ROCPlot
from ..plotting.regression import CorrelationPlot


class ModelRetriever(ModelDataSetsMixIn):
    def getModel(
        self,
        dataset: QSPRDataset,
        name: str,
        alg: Type = RandomForestClassifier
    ) -> SklearnModel:
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
            data=dataset,
            base_dir=self.generatedModelsPath,
            alg=alg,
        )


class ROCPlotTest(ModelRetriever, TestCase):
    """Test ROC curve plotting class."""
    def testPlotSingle(self):
        """Test plotting ROC curve for single task."""
        dataset = self.createLargeTestDataSet(
            "test_roc_plot_single_data",
            target_props=[{
                "name": "CL",
                "task": TargetTasks.SINGLECLASS,
                "th": [6.5]
            }],
            preparation_settings=self.getDefaultPrep(),
        )
        model = self.getModel(dataset, "test_roc_plot_single_model")
        score_func = "roc_auc_ovr" if model.task.isMultiTask() else "roc_auc"
        CrossValAssessor(scoring = score_func)(model)
        TestSetAssessor(scoring = score_func)(model)
        model.save()
        # make plots
        plt = ROCPlot([model])
        # cross validation plot
        ax = plt.make("CL_class", validation="cv")[0]
        self.assertIsInstance(ax, Figure)
        self.assertTrue(os.path.exists(f"{model.outPrefix}.cv.png"))
        # independent test set plot
        ax = plt.make("CL_class", validation="ind")[0]
        self.assertIsInstance(ax, Figure)
        self.assertTrue(os.path.exists(f"{model.outPrefix}.ind.png"))


class MetricsPlotTest(ModelRetriever, TestCase):
    """Test metrics plotting class."""
    def testPlotSingle(self):
        """Test plotting metrics for single task."""
        dataset = self.createLargeTestDataSet(
            "test_metrics_plot_single_data",
            target_props=[{
                "name": "CL",
                "task": TargetTasks.SINGLECLASS,
                "th": [6.5]
            }],
            preparation_settings=self.getDefaultPrep(),
        )
        model = self.getModel(dataset, "test_metrics_plot_single_model")
        score_func = "roc_auc"
        CrossValAssessor(scoring = score_func)(model)
        TestSetAssessor(scoring = score_func)(model)
        model.save()
        # generate metrics plot and associated files
        plt = MetricsPlot([model])
        figures, summary = plt.make("CL_class", out_dir=model.outDir)
        for fig, ax in figures:
            self.assertIsInstance(fig, Figure)
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertTrue(os.path.exists(f"{model.outDir}/metrics_summary.tsv"))
        self.assertTrue(os.path.exists(f"{model.outDir}/metrics_f1_score.png"))


class CorrPlotTest(ModelRetriever, TestCase):
    """Test correlation plotting class."""
    def testPlotSingle(self):
        """Test plotting correlation for single task."""
        dataset = self.createLargeTestDataSet(
            "test_corr_plot_single_data", preparation_settings=self.getDefaultPrep()
        )
        model = self.getModel(
            dataset, "test_corr_plot_single_model", alg=RandomForestRegressor
        )
        score_func = "r2"
        CrossValAssessor(scoring = score_func)(model)
        TestSetAssessor(scoring = score_func)(model)
        model.save()
        # generate metrics plot and associated files
        plt = CorrelationPlot([model])
        axes, summary = plt.make("CL", out_dir=model.outDir)
        self.assertIsInstance(summary, pd.DataFrame)
        for ax in axes:
            self.assertIsInstance(ax, SubplotBase)
        self.assertTrue(os.path.exists(f"{model.outDir}/corrplot.png"))
