"""Tests for plotting module."""

import os
from typing import Type
from unittest import TestCase

import pandas as pd
from matplotlib.axes import SubplotBase
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..data.data import QSPRDataset
from ..models.models import QSPRsklearn
from ..models.tasks import TargetTasks
from ..models.tests import ModelDataSetsMixIn
from ..plotting.classification import MetricsPlot, ROCPlot
from ..plotting.regression import CorrelationPlot


class ROCPlotTest(ModelDataSetsMixIn, TestCase):
    """Test ROC curve plotting class."""

    @staticmethod
    def getModel(
            dataset: QSPRDataset,
            name: str,
            alg: Type = RandomForestClassifier
    ) -> QSPRsklearn:
        """Get a model for testing.

        Args:
            dataset (QSPRDataset):
                Dataset to use for model.
            name (str):
                Name of model.
            alg (Type, optional):
                Algorithm to use for model. Defaults to `RandomForestClassifier`.

        Returns:
            QSPRsklearn:
                The new model.

        """
        return QSPRsklearn(
            name=name,
            data=dataset,
            base_dir=os.path.dirname(
                ROCPlotTest.qsprModelsPath.replace("qspr/models", "")
            ),
            alg=alg,
        )

    def testPlotSingle(self):
        """Test plotting ROC curve for single task."""
        dataset = self.create_large_dataset(
            "test_roc_plot_single_data",
            target_props=[{
                "name": "CL",
                "task": TargetTasks.SINGLECLASS,
                "th": [50]
            }],
            preparation_settings=self.get_default_prep(),
        )
        model = self.getModel(dataset, "test_roc_plot_single_model")
        model.evaluate(save=True)
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


class MetricsPlotTest(ModelDataSetsMixIn, TestCase):
    """Test metrics plotting class."""
    @staticmethod
    def getModel(
            dataset: QSPRDataset,
            name: str,
            alg: Type = RandomForestClassifier
    ) -> QSPRsklearn:
        """Get a model for testing.

        Args:
            dataset (QSPRDataset):
                Dataset to use for model.
            name (str):
                Name of model.
            alg (Type, optional):
                Algorithm to use for model. Defaults to `RandomForestClassifier`.
        """
        return QSPRsklearn(
            name=name,
            data=dataset,
            base_dir=os.path.dirname(
                ROCPlotTest.qsprModelsPath.replace("qspr/models", "")
            ),
            alg=alg,
        )

    def testPlotSingle(self):
        """Test plotting metrics for single task."""
        dataset = self.create_large_dataset(
            "test_metrics_plot_single_data",
            target_props=[{
                "name": "CL",
                "task": TargetTasks.SINGLECLASS,
                "th": [50]
            }],
            preparation_settings=self.get_default_prep(),
        )
        model = self.getModel(dataset, "test_metrics_plot_single_model")
        model.evaluate(save=True)
        model.save()
        # generate metrics plot and associated files
        plt = MetricsPlot([model])
        figures, summary = plt.make("CL_class", out_dir=model.outDir)
        for fig, ax in figures:
            self.assertIsInstance(fig, Figure)
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertTrue(os.path.exists(f"{model.outDir}/metrics_summary.tsv"))
        self.assertTrue(os.path.exists(f"{model.outDir}/metrics_f1_score.png"))


class CorrPlotTest(ModelDataSetsMixIn, TestCase):
    """Test correlation plotting class."""

    @staticmethod
    def getModel(
            dataset: QSPRDataset,
            name: str,
            alg: Type = RandomForestRegressor
    ) -> QSPRsklearn:
        """Get a model for testing.

        Args:
            dataset (QSPRDataset):
                Dataset to use for model.
            name (str):
                Name of model.
            alg (Type, optional):
                Algorithm to use for model. Defaults to `RandomForestRegressor`.
        """
        return QSPRsklearn(
            name=name,
            data=dataset,
            base_dir=os.path.dirname(
                ROCPlotTest.qsprModelsPath.replace("qspr/models", "")
            ),
            alg=alg,
        )

    def testPlotSingle(self):
        """Test plotting correlation for single task."""
        dataset = self.create_large_dataset(
            "test_corr_plot_single_data", preparation_settings=self.get_default_prep()
        )
        model = self.getModel(dataset, "test_corr_plot_single_model")
        model.evaluate(save=True)
        model.save()
        # generate metrics plot and associated files
        plt = CorrelationPlot([model])
        axes, summary = plt.make("CL", out_dir=model.outDir)
        self.assertIsInstance(summary, pd.DataFrame)
        for ax in axes:
            self.assertIsInstance(ax, SubplotBase)
        self.assertTrue(os.path.exists(f"{model.outDir}/corrplot.png"))
