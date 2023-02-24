"""Plotting functions for classification models."""
import os.path
from abc import ABC
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from qsprpred.metrics.calibration import calibration_error
from qsprpred.models.interfaces import QSPRModel
from qsprpred.models.tasks import ModelTasks, TargetTasks
from qsprpred.plotting.interfaces import ModelPlot
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    auc,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


class ClassifierPlot(ModelPlot, ABC):
    """Base class for plots of classification models."""

    def getSupportedTasks(self):
        """Return a list of tasks supported by this plotter."""
        return [ModelTasks.SINGLECLASS]


class ROCPlot(ClassifierPlot):
    """Plot of ROC-curve (receiver operating characteristic curve) for a given model."""

    def makeCV(self, model: QSPRModel):
        """Make the plot for a given model using cross-validation data.

        Args:
            model (QSPRModel): the model to plot the data from.

        Returns:
           ax (matplotlib.axes.Axes): the axes object containing the plot.
        """
        df = pd.read_table(self.cvPaths[model])

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        ax = plt.gca()
        for fold in df.Fold.unique():
            # get labels
            y_pred = df.Score[df.Fold == fold]
            y_true = df.Label[df.Fold == fold]

            # do plotting
            viz = RocCurveDisplay.from_predictions(
                y_true,
                y_pred,
                name="ROC fold {}".format(fold + 1),
                alpha=0.3,
                lw=1,
                ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=f"Receiver Operating Characteristic ({self.modelNames[model]})",
        )
        ax.legend(loc="lower right")
        return ax

    def makeInd(self, model: QSPRModel):
        """Make the plot for a given model using independent test data.

        Args:
            model (QSPRModel): the model to plot the data from.

        Returns:
              ax (matplotlib.axes.Axes): the axes object containing the plot.
        """
        df = pd.read_table(self.indPaths[model])
        y_pred = df.Score
        y_true = df.Label

        ax = plt.gca()
        RocCurveDisplay.from_predictions(
            y_true,
            y_pred,
            name="ROC",
            ax=ax,
        )
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance")
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=f"Receiver Operating Characteristic ({self.modelNames[model]})",
        )
        ax.legend(loc="lower right")
        return ax

    def make(self, validation: str = "cv", figsize: tuple = (6, 6), save: bool = True, show: bool = False):
        """Make the plot for a given validation type. Displays the plot and optionally saves it to a file.

        Args:
            validation (str): The type of validation data to use.
                              Can be either 'cv' for cross-validation or 'ind' for independent test set.
            figsize (tuple): The size of the figure to create.
            save (bool): Whether to save the plot to a file.
            show (bool): Whether to display the plot.

        Returns:
            axes (list): A list of matplotlib axes objects containing the plots.
        """
        choices = {
            "cv": self.makeCV,
            "ind": self.makeInd
        }
        axes = []
        for model in self.models:
            fig, ax = plt.subplots(figsize=figsize)
            ax = choices[validation](model)
            axes.append(fig)
            if save:
                fig.savefig(f'{self.modelOuts[model]}.{validation}.png')
            if show:
                plt.show()
                plt.clf()
        return axes


class PRCPlot(ClassifierPlot):
    """Plot of Precision-Recall curve for a given model."""

    def makeCV(self, model: QSPRModel):
        """Make the plot for a given model using cross-validation data.

        Args:
            model (QSPRModel): the model to plot the data from.

        Returns:
           ax (matplotlib.axes.Axes): the axes object containing the plot.
        """
        df = pd.read_table(self.cvPaths[model])

        y_real = []
        y_predproba = []

        ax = plt.gca()
        for fold in df.Fold.unique():
            # get labels
            y_pred = df.Score[df.Fold == fold]
            y_true = df.Label[df.Fold == fold]
            y_predproba.append(y_pred)
            y_real.append(y_true)
            # do plotting
            viz = PrecisionRecallDisplay.from_predictions(
                y_true,
                y_pred,
                name="PRC fold {}".format(fold + 1),
                ax=ax,
                alpha=0.3,
                lw=1,
            )
        # Linear iterpolation of PR curve is not recommended, so we don't plot "chance"
        # https://dl.acm.org/doi/10.1145/1143844.1143874

        # Plotting the average precision-recall curve over the cross validation runs
        y_real = np.concatenate(y_real)
        y_predproba = np.concatenate(y_predproba)
        viz = PrecisionRecallDisplay.from_predictions(
            y_real,
            y_predproba,
            name="Mean PRC",
            color="b",
            ax=ax,
            lw=1.2,
            alpha=0.8,
        )
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=f"Precision-Recall Curve ({self.modelNames[model]})",
        )
        ax.legend(loc="best")
        return ax

    def makeInd(self, model: QSPRModel):
        """Make the plot for a given model using independent test data.

        Args:
            model (QSPRModel): the model to plot the data from.

        Returns:
              ax (matplotlib.axes.Axes): the axes object containing the plot.
        """
        df = pd.read_table(self.indPaths[model])
        y_pred = df.Score
        y_true = df.Label

        ax = plt.gca()
        PrecisionRecallDisplay.from_predictions(
            y_true,
            y_pred,
            name="PRC",
            ax=ax,
        )
        #
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=f"Receiver Operating Characteristic ({self.modelNames[model]})",
        )
        ax.legend(loc="best")
        return ax

    def make(self, validation: str = "cv", figsize: tuple = (6, 6), save: bool = True, show: bool = False):
        """Make the plot for a given validation type. Displays the plot and optionally saves it to a file.

        Args:
            validation (str): The type of validation data to use.
                              Can be either 'cv' for cross-validation or 'ind' for independent test set.
            figsize (tuple): The size of the figure to create.
            save (bool): Whether to save the plot to a file.
            show (bool): Whether to display the plot.

        Returns:
            axes (list): A list of matplotlib axes objects containing the plots.
        """
        choices = {
            "cv": self.makeCV,
            "ind": self.makeInd
        }
        axes = []
        for model in self.models:
            fig, ax = plt.subplots(figsize=figsize)
            ax = choices[validation](model)
            axes.append(ax)
            if save:
                fig.savefig(f'{self.modelOuts[model]}.{validation}.png')
            if show:
                plt.show()
                plt.clf()
        return axes


class CalibrationPlot(ClassifierPlot):
    """Plot of calibration curve for a given model."""

    def makeCV(self, model: QSPRModel, n_bins: int = 10):
        """Make the plot for a given model using cross-validation data.

        Args:
            model (QSPRModel): the model to plot the data from.
            n_bins (int): The number of bins to use for the calibration curve.

        Returns:
            ax (matplotlib.axes.Axes): the axes object containing the plot.
        """
        df = pd.read_table(self.cvPaths[model])

        y_real = []
        y_predproba = []

        ax = plt.gca()
        for fold in df.Fold.unique():
            # get labels
            y_pred = df.Score[df.Fold == fold]
            y_true = df.Label[df.Fold == fold]
            y_predproba.append(y_pred)
            y_real.append(y_true)
            # do plotting
            viz = CalibrationDisplay.from_predictions(
                y_true,
                y_pred,
                n_bins=n_bins,
                name="Fold: {}".format(fold + 1),
                ax=ax,
                alpha=0.3,
                lw=1,
            )

        # Plotting the average precision-recall curve over the cross validation runs
        y_real = np.concatenate(y_real)
        y_predproba = np.concatenate(y_predproba)
        viz = CalibrationDisplay.from_predictions(
            y_real,
            y_predproba,
            n_bins=n_bins,
            name="Mean",
            color="b",
            ax=ax,
            lw=1.2,
            alpha=0.8,
        )
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=f"Calibration Curve ({self.modelNames[model]})",
        )
        ax.legend(loc="best")
        return ax

    def makeInd(self, model: QSPRModel, n_bins: int = 10):
        """Make the plot for a given model using independent test data.

        Args:
            model (QSPRModel): the model to plot the data from.
            n_bins (int): The number of bins to use for the calibration curve.

        Returns:
            ax (matplotlib.axes.Axes): the axes object containing the plot.
        """
        df = pd.read_table(self.indPaths[model])
        y_pred = df.Score
        y_true = df.Label

        ax = plt.gca()
        CalibrationDisplay.from_predictions(
            y_true,
            y_pred,
            n_bins=n_bins,
            name="Calibration",
            ax=ax,
        )
        #
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=f"Calibration Curve ({self.modelNames[model]})",
        )
        ax.legend(loc="best")
        return ax

    def make(self, validation: str = "cv", n_bins: int = 10,
             figsize: tuple = (6, 6), save: bool = True, show: bool = False):
        """Make the plot for a given validation type. Displays the plot and optionally saves it to a file.

        Args:
            validation (str): The type of validation data to use.
                              Can be either 'cv' for cross-validation or 'ind' for independent test set.
            n_bins (int): The number of bins to use for the calibration curve.
            figsize (tuple): The size of the figure to create.
            save (bool): Whether to save the plot to a file.
            show (bool): Whether to display the plot.

        Returns:
            axes (list): A list of matplotlib axes objects containing the plots.
        """
        choices = {
            "cv": self.makeCV,
            "ind": self.makeInd
        }
        axes = []
        for model in self.models:
            fig, ax = plt.subplots(figsize=figsize)
            ax = choices[validation](model, n_bins)
            axes.append(ax)
            if save:
                fig.savefig(f'{self.modelOuts[model]}.{validation}.png')
            if show:
                plt.show()
                plt.clf()
        return axes


class MetricsPlot(ClassifierPlot):
    """Plot of metrics for a given model.

    Includes the following metrics:
            f1_score, matthews_corrcoef, precision_score, recall_score, accuracy_score, calibration_error

    Attributes:
        metrics (list): A list of metrics to plot.
        decision (float): The decision threshold to use for the metrics.
        summary (dict): A dictionary containing the data to plot.
    """

    def __init__(self,
                 models: List[QSPRModel],
                 metrics: List[callable] = (
                     f1_score,
                     matthews_corrcoef,
                     precision_score,
                     recall_score,
                     accuracy_score,
                     calibration_error
                 ),
                 decision_threshold: float = 0.5):
        """Initialise the metrics plot.

        Args:
            models (list): A list of QSPRModel objects to plot the data from.
            metrics (list): A list of metrics to plot.
            decision_threshold (float): The decision threshold to use for the metrics.
        """
        super().__init__(models)
        self.metrics = metrics
        self.decision = decision_threshold
        self.summary = None
        self.reset()

    def reset(self):
        """Reset the summary data."""
        self.summary = {
            'Metric': [],
            'Model': [],
            'TestSet': [],
            'Value': []
        }

    def make(self, save: bool = True, show: bool = False, filename_prefix: str = 'metrics', out_dir: str = "."):
        """Make the plot for a given validation type. Displays the plot and optionally saves it to a file.

        Args:
            save (bool): Whether to save the plot to a file.
            show (bool): Whether to display the plot.
            filename_prefix (str): The prefix to use for the filename.
            out_dir (str): The directory to save the plot to.
        """
        self.reset()
        for model in self.models:
            df = pd.read_table(self.cvPaths[model])

            for fold in df.Fold.unique():
                y_pred = df.Score[df.Fold == fold]
                y_pred_values = [1 if x > self.decision else 0 for x in y_pred]
                y_true = df.Label[df.Fold == fold]
                for metric in self.metrics:
                    val = metric(y_true, y_pred_values)
                    self.summary['Metric'].append(metric.__name__)
                    self.summary['Model'].append(self.modelNames[model])
                    self.summary['TestSet'].append(f'CV{fold + 1}')
                    self.summary['Value'].append(val)

            df = pd.read_table(self.indPaths[model])
            y_pred = df.Score
            y_pred_values = [1 if x > 0.5 else 0 for x in y_pred]
            y_true = df.Label
            for metric in self.metrics:
                val = metric(y_true, y_pred_values)
                self.summary['Metric'].append(metric.__name__)
                self.summary['Model'].append(self.modelNames[model])
                self.summary['TestSet'].append('IND')
                self.summary['Value'].append(val)

        df_summary = pd.DataFrame(self.summary)
        if save:
            df_summary.to_csv(
                os.path.join(
                    out_dir,
                    f"{filename_prefix}_summary.tsv"),
                sep='\t',
                index=False,
                header=True)

        figures = []
        for metric in df_summary.Metric.unique():
            df_metric = df_summary[df_summary.Metric == metric]
            cv_avg = df_metric[df_metric.TestSet != 'IND'][['Model', 'Value']].groupby('Model').aggregate(np.mean)
            cv_std = df_metric[df_metric.TestSet != 'IND'][['Model', 'Value']].groupby('Model').aggregate(np.std)
            ind_vals = df_metric[df_metric.TestSet == 'IND'][['Model', 'Value']].groupby('Model').aggregate(np.sum)

            models = cv_avg.index

            x = np.arange(len(models))  # the label locations
            width = 0.35  # the width of the bars

            fig, ax = plt.subplots(figsize=(7, 10))
            rects1 = ax.bar(
                x - width / 2,
                cv_avg.Value,
                width,
                label='CV',
                yerr=cv_std.Value,
                ecolor='black',
                capsize=5)
            rects2 = ax.bar(x + width / 2, ind_vals.Value, width, label='Test')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} by Model and Test Set')
            ax.set_xticks(x, models)
            ax.legend()

            # ax.bar_label(rects1, padding=3)
            # ax.bar_label(rects2, padding=3)

            fig.tight_layout()
            plt.xticks(rotation=75)
            plt.ylim([0, 1.3])
            plt.subplots_adjust(bottom=0.4)
            plt.axhline(y=1.0, color='grey', linestyle='-', alpha=0.3)
            if save:
                plt.savefig(os.path.join(out_dir, f"{filename_prefix}_{metric}.png"))
            if show:
                plt.show()
                plt.clf()
            figures.append((fig, ax))
        return figures, df_summary
