"""
tests

Created by: Martin Sicho
On: 12.02.23, 17:02
"""
import os
from unittest import TestCase

import pandas as pd
from matplotlib.axes import SubplotBase
from matplotlib.figure import Figure
from qsprpred.models.models import QSPRsklearn
from qsprpred.models.tasks import TargetTasks
from qsprpred.models.tests import ModelDataSetsMixIn
from qsprpred.plotting.classification import MetricsPlot, ROCPlot
from qsprpred.plotting.regression import CorrelationPlot
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class ROCPlotTest(ModelDataSetsMixIn, TestCase):

    @staticmethod
    def get_model(dataset, name, alg=RandomForestClassifier):
        return QSPRsklearn(
            name=name,
            data=dataset,
            base_dir=os.path.dirname(ROCPlotTest.qsprmodelspath.replace("qspr/models", "")),
            alg=alg,
        )

    def test_plot_single(self):
        dataset = self.create_large_dataset(
            "test_roc_plot_single_data",
            target_props=[{"name": "CL", "task": TargetTasks.SINGLECLASS, "th": [50]}],
            preparation_settings=self.get_default_prep())
        model = self.get_model(dataset, "test_roc_plot_single_model")
        model.evaluate(save=True)
        model.save()

        plt = ROCPlot([model])

        # cross validation plot
        ax = plt.make("CL_class", validation='cv')[0]
        self.assertIsInstance(ax, Figure)
        self.assertTrue(os.path.exists(f"{model.outPrefix}.cv.png"))

        # independent test set plot
        ax = plt.make("CL_class", validation='ind')[0]
        self.assertIsInstance(ax, Figure)
        self.assertTrue(os.path.exists(f"{model.outPrefix}.ind.png"))


class MetricsPlotTest(ModelDataSetsMixIn, TestCase):

    @staticmethod
    def get_model(dataset, name, alg=RandomForestClassifier):
        return QSPRsklearn(
            name=name,
            data=dataset,
            base_dir=os.path.dirname(ROCPlotTest.qsprmodelspath.replace("qspr/models", "")),
            alg=alg,
        )

    def test_plot_single(self):
        dataset = self.create_large_dataset(
            "test_metrics_plot_single_data",
            target_props=[{"name": "CL", "task": TargetTasks.SINGLECLASS, "th": [50]}],
            preparation_settings=self.get_default_prep())
        model = self.get_model(dataset, "test_metrics_plot_single_model")
        model.evaluate(save=True)
        model.save()

        plt = MetricsPlot([model])

        # generate metrics plot and associated files
        figures, summary = plt.make("CL_class", out_dir=model.outDir)
        for fig, ax in figures:
            self.assertIsInstance(fig, Figure)
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertTrue(os.path.exists(f"{model.outDir}/metrics_summary.tsv"))
        self.assertTrue(os.path.exists(f"{model.outDir}/metrics_f1_score.png"))


class CorrPlotTest(ModelDataSetsMixIn, TestCase):

    @staticmethod
    def get_model(dataset, name, alg=RandomForestRegressor):
        return QSPRsklearn(
            name=name,
            data=dataset,
            base_dir=os.path.dirname(ROCPlotTest.qsprmodelspath.replace("qspr/models", "")),
            alg=alg,
        )

    def test_plot_single(self):
        dataset = self.create_large_dataset("test_corr_plot_single_data", preparation_settings=self.get_default_prep())
        model = self.get_model(dataset, "test_corr_plot_single_model")
        model.evaluate(save=True)
        model.save()

        plt = CorrelationPlot([model])

        # generate metrics plot and associated files
        axes, summary = plt.make("CL", out_dir=model.outDir)
        self.assertIsInstance(summary, pd.DataFrame)
        for ax in axes:
            self.assertIsInstance(ax, SubplotBase)
        self.assertTrue(os.path.exists(f"{model.outDir}/corrplot.png"))
