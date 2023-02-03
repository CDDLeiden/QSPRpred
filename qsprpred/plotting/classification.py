"""
classification

Created by: Martin Sicho
On: 16.11.22, 12:12
"""
import os
from abc import ABC
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay, auc, f1_score, matthews_corrcoef, precision_score, recall_score, \
    accuracy_score

from qsprpred.models.interfaces import QSPRModel
from qsprpred.plotting.interfaces import ModelPlot


class ClassifierPlot(ModelPlot, ABC):

    def getSupportedTypes(self):
        return ["CLS"]

class ROCPlot(ClassifierPlot):

    def makeCV(self, model : QSPRModel):
        df = pd.read_table(self.cvPaths[model])

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()
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

    def makeInd(self, model : QSPRModel):
        df = pd.read_table(self.indPaths[model])
        y_pred = df.Score
        y_true = df.Label

        fig, ax = plt.subplots()
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

    def make(self, validation : str = "cv", save : bool = True, show : bool = False):
        """
        Make the plot for a given validation type. Displays the plot and optionally saves it to a file.
        """

        choices = {
            "cv": self.makeCV,
            "ind": self.makeInd
        }
        figures = []
        for model in self.models:
            fig = choices[validation](model)
            figures.append(fig)
            if save:
                plt.savefig(f'{self.modelOuts[model]}.{validation}.png')
            if show:
                plt.show()
                plt.clf()
        return figures

class MetricsPlot(ClassifierPlot):

    def __init__(self,
        models : List[QSPRModel],
        metrics : List[callable] = (
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            accuracy_score
        ),
        decision_threshold : float = 0.5)\
        :
        super().__init__(models)
        self.metrics = metrics
        self.decision = decision_threshold
        self.summary = None
        self.reset()

    def reset(self):
        self.summary = {
            'Metric': [],
            'Model': [],
            'TestSet': [],
            'Value': []
        }

    def make(self, save : bool = True, show : bool = False, filename_prefix : str = 'metrics', save_summary_to : str = None):
        """
        Make the plot for a given validation type. Displays the plot and optionally saves it to a file.
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
        if save_summary_to is not None:
            df_summary.to_csv(save_summary_to, sep='\t', index=False, header=True)

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
            rects1 = ax.bar(x - width / 2, cv_avg.Value, width, label='CV', yerr=cv_std.Value, ecolor='black', capsize=5)
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
                plt.savefig(f'{filename_prefix}_{metric}.png')
            if show:
                plt.show()
                plt.clf()
            figures.append((fig, ax))
        return figures, df_summary
