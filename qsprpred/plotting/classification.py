"""Plotting functions for classification models."""
import os.path
import re
from abc import ABC
from copy import deepcopy
from typing import List, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
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
    roc_auc_score,
    confusion_matrix,
)

from ..models.metrics import calibration_error
from ..models.model import QSPRModel
from ..plotting.base_plot import ModelPlot
from ..tasks import ModelTasks


class ClassifierPlot(ModelPlot, ABC):
    """Base class for plots of classification models."""

    def getSupportedTasks(self) -> List[ModelTasks]:
        """Return a list of tasks supported by this plotter."""
        return [
            ModelTasks.SINGLECLASS,
            ModelTasks.MULTICLASS,
            ModelTasks.MULTITASK_SINGLECLASS,
            ModelTasks.MULTITASK_MULTICLASS,
        ]

    def prepareAssessment(self, assessment_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare assessment dataframe for plotting

        Args:
            assessment_df (pd.DataFrame):
                the assessment dataframe containing the experimental and predicted
                values for each property. The dataframe should have the following
                columns:
                QSPRID, Fold (opt.), <property_name>_<suffixes>_<Label/Prediction/ProbabilityClass_X>

        Returns:
            pd.DataFrame:
                The dataframe containing the assessment results,
                columns: QSPRID, Fold, Property, Label, Prediction, Class, Set
        """
        # change all property columns into one column
        id_vars = ["QSPRID", "Fold"] if "Fold" in assessment_df.columns else ["QSPRID"]
        df = assessment_df.melt(id_vars=id_vars)
        # split the variable (<property_name>_<suffixes>_<Label/Prediction/ProbabilityClass_X>) column
        # into the property name and the type (Label or Prediction or ProbabilityClass_X)
        pattern = re.compile(
            r"^(?P<Property>.*?)_(?P<type>Label|Prediction|ProbabilityClass_\d+)$"
        )
        df[["Property", "type"]] = df["variable"].apply(
            lambda x: pd.Series(pattern.match(x).groupdict())
        )
        df.drop("variable", axis=1, inplace=True)
        # pivot the dataframe so that Label and Prediction are separate columns
        df = df.pivot_table(
            index=[*id_vars, "Property"], columns="type", values="value"
        )
        df.reset_index(inplace=True)
        df.columns.name = None
        df["Label"] = df["Label"].astype(int)
        df["Prediction"] = df["Prediction"].astype(int)
        # Add Fold column if it doesn't exist (for independent test set)
        if "Fold" not in df.columns:
            df["Fold"] = "Independent Test"
            df["Set"] = "Independent Test"
        else:
            df["Set"] = "Cross Validation"
        return df

    def prepareClassificationResults(self) -> pd.DataFrame:
        """Prepare classification results dataframe for plotting.

        Returns:
            pd.DataFrame:
                the dataframe containing the classficiation results,
                columns: Model, QSPRID, Fold, Property, Label, Prediction, Set
        """
        model_results = {}
        for m, model in enumerate(self.models):
            # Read in and prepare the cross-validation and independent test set results
            df_cv = self.prepareAssessment(pd.read_table(self.cvPaths[model]))
            df_ind = self.prepareAssessment(pd.read_table(self.indPaths[model]))
            # concatenate the cross-validation and independent test set results
            df = pd.concat([df_cv, df_ind])
            print(model.name)
            model_results[model.name] = df
        # concatenate the results from all models and add the model name as a column
        df = (
            pd.concat(
                model_results.values(), keys=model_results.keys(), names=["Model"]
            )
            .reset_index(level=1, drop=True)
            .reset_index()
        )

        self.results = df
        return df

    def calculateMultiClassMetrics(self, df, average_type, n_classes):
        """Calculate metrics for a given dataframe."""
        # check if ProbabilityClass_X columns exist
        proba = all([f"ProbabilityClass_{i}" in df.columns for i in range(n_classes)])

        if average_type == "All":
            metrics = {
                "accuracy": accuracy_score(df.Label, df.Prediction),
                "matthews_corrcoef": matthews_corrcoef(df.Label, df.Prediction),
            }
            if proba:
                metrics["calibration_error"] = calibration_error(
                    df.Label, df[[f"ProbabilityClass_{i}" for i in range(n_classes)]]
                )
            return pd.Series(metrics)

        # check if average type is int (i.e. class number, so we can calculate metrics for a single class)
        if isinstance(average_type, int):
            class_num = average_type
            average_type = None

        metrics = {
            "precision": precision_score(df.Label, df.Prediction, average=average_type),
            "recall": recall_score(df.Label, df.Prediction, average=average_type),
            "f1": f1_score(df.Label, df.Prediction, average=average_type),
        }
        if proba:
            metrics["roc_auc_ovr"] = roc_auc_score(
                df.Label,
                df[[f"ProbabilityClass_{i}" for i in range(n_classes)]],
                multi_class="ovr",
                average=average_type,
            )

        # FIXME: metrics are only returned for class "class_num", but calculated for all classes
        # as returning a list of metrics for each class gives a dataframe with lists as values
        # which is difficult to explode. Need to find a better way to do this.
        if average_type is None:
            metrics = {k: v[class_num] for k, v in metrics.items()}

        # Conditionally include roc_auc_ovo for non-micro averages
        if average_type != "micro" and average_type is not None and proba:
            metrics["roc_auc_ovo"] = roc_auc_score(
                df.Label,
                df[[f"ProbabilityClass_{i}" for i in range(n_classes)]],
                multi_class="ovo",
                average=average_type,
            )

        return pd.Series(metrics)

    def calculateSingleClassMetrics(self, df):
        """Calculate metrics for a given dataframe."""

        # check if ProbabilityClass_1 column exists
        proba = "ProbabilityClass_1" in df.columns

        metrics = {
            "accuracy": accuracy_score(df.Label, df.Prediction),
            "precision": precision_score(df.Label, df.Prediction),
            "recall": recall_score(df.Label, df.Prediction),
            "f1": f1_score(df.Label, df.Prediction),
            "matthews_corrcoef": matthews_corrcoef(df.Label, df.Prediction),
        }

        if proba:
            metrics["calibration_error"] = calibration_error(
                df.Label, df.ProbabilityClass_1
            )
            metrics["roc_auc"] = roc_auc_score(df.Label, df.ProbabilityClass_1)

        return pd.Series(metrics)

    def getSummary(self):
        """Get summary statistics for classification results."""
        if not hasattr(self, "results"):
            self.prepareClassificationResults()

        df = deepcopy(self.results)

        summary_list = {}

        # make summary for each model and property
        for model_name in df.Model.unique():
            for property_name in df.Property.unique():
                df_subset = df[
                    (df.Model == model_name) & (df.Property == property_name)
                ]

                # get the number of classes
                n_classes = df_subset["Label"].nunique()

                # calculate metrics for binary and multi-class properties
                if n_classes == 2:
                    summary_list[f"{model_name}_{property_name}_Binary"] = (
                        df_subset.groupby(["Model", "Fold", "Property"]).apply(
                            lambda x: self.calculateSingleClassMetrics(x)
                        )
                    ).reset_index()
                    summary_list[f"{model_name}_{property_name}_Binary"][
                        "Class"
                    ] = "Binary"
                else:
                    # calculate metrics for each class, average type and non-average type metrics
                    class_list = [
                        *["macro", "micro", "weighted", "All"],
                        *list(range(n_classes)),
                    ]

                    for class_type in class_list:
                        summary_list[f"{model_name}_{property_name}_{class_type}"] = (
                            df_subset.groupby(["Model", "Fold", "Property"]).apply(
                                lambda x: self.calculateMultiClassMetrics(
                                    x, class_type, n_classes
                                )
                            )
                        ).reset_index()
                        summary_list[f"{model_name}_{property_name}_{class_type}"][
                            "Class"
                        ] = class_type

        df_summary = pd.concat(summary_list.values(), ignore_index=True)

        df_summary["Set"] = df_summary["Fold"].apply(
            lambda x: "Independent Test"
            if x == "Independent Test"
            else "Cross Validation"
        )
        self.summary = df_summary
        return df_summary


class ROCPlot(ClassifierPlot):
    """Plot of ROC-curve (receiver operating characteristic curve)
    for a given classification model.
    """

    def getSupportedTasks(self) -> List[ModelTasks]:
        """Return a list of tasks supported by this plotter."""
        return [ModelTasks.SINGLECLASS, ModelTasks.MULTITASK_SINGLECLASS]

    def makeCV(self, model: QSPRModel, property_name: str) -> plt.Axes:
        """Make the plot for a given model using cross-validation data.

        Many thanks to the scikit-learn documentation since the code below
        borrows heavily from the example at:

        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

        Args:
            model (QSPRModel):
                the model to plot the data from.
            property_name (str):
                name of the property to plot (should correspond to the prefix
                of the column names in the data files).

        Returns:
           ax (matplotlib.axes.Axes): the axes object containing the plot.
        """
        df = pd.read_table(self.cvPaths[model])
        # get true positive rate and false positive rate for each fold
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        ax = plt.gca()
        for fold in df.Fold.unique():
            # get labels
            y_pred = df[f"{property_name}_ProbabilityClass_1"][df.Fold == fold]
            y_true = df[f"{property_name}_Label"][df.Fold == fold]
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
        # plot chance line
        ax.plot(
            [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
        )
        # plot mean ROC across folds
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
        # plot standard deviation across folds
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
        # set axes limits and labels
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=f"Receiver Operating Characteristic ({self.modelNames[model]})",
        )
        ax.legend(loc="lower right")
        return ax

    def makeInd(self, model: QSPRModel, property_name: str) -> plt.Axes:
        """Make the ROC plot for a given model using independent test data.

        Args:
            model (QSPRModel):
                the model to plot the data from.
            property_name (str):
                name of the property to plot
                (should correspond to the prefix of the column names in the data files).

        Returns:
              ax (matplotlib.axes.Axes): the axes object containing the plot.
        """
        df = pd.read_table(self.indPaths[model])
        y_pred = df[f"{property_name}_ProbabilityClass_1"]
        y_true = df[f"{property_name}_Label"]

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

    def make(
        self,
        save: bool = True,
        show: bool = False,
        property_name: str | None = None,
        validation: str = "cv",
        fig_size: tuple = (6, 6),
    ) -> list[plt.Axes]:
        """Make the ROC plot for given validation sets.

        Args:
            property_name (str):
                name of the predicted property to plot (should correspond to the
                prefix of the column names in `cvPaths` or `indPaths` files).
                If `None`, the first property in the model's `targetProperties` list
                will be used.
            validation (str):
                The type of validation set to read data for. Can be either 'cv'
                for cross-validation or 'ind' for independent test set.
            fig_size (tuple):
                The size of the figure to create.
            save (bool):
                Whether to save the plot to a file.
            show (bool):
                Whether to display the plot.

        Returns:
            axes (list[plt.Axes]):
                A list of matplotlib axes objects containing the plots.
        """
        if property_name is None:
            property_name = self.models[0].targetProperties[0].name
        # fetch the correct plotting function based on validation set type
        # and make the plot for each model
        choices = {"cv": self.makeCV, "ind": self.makeInd}
        axes = []
        for model in self.models:
            fig, ax = plt.subplots(figsize=fig_size)
            choices[validation](model, property_name)
            axes.append(fig)
            if save:
                fig.savefig(f"{self.modelOuts[model]}.{validation}.png")
            if show:
                plt.show()
                plt.clf()
        return axes


class PRCPlot(ClassifierPlot):
    """Plot of Precision-Recall curve for a given model."""

    def getSupportedTasks(self) -> List[ModelTasks]:
        """Return a list of tasks supported by this plotter."""
        return [ModelTasks.SINGLECLASS, ModelTasks.MULTITASK_SINGLECLASS]

    def makeCV(self, model: QSPRModel, property_name: str) -> plt.Axes:
        """Make the plot for a given model using cross-validation data.

        Args:
            model (QSPRModel):
                the model to plot the data from.
            property_name (str):
                name of the property to plot
                (should correspond to the prefix of the column names in the data files).

        Returns:
                ax (matplotlib.axes.Axes):
                    the axes object containing the plot.
        """
        # read data from file for each fold
        df = pd.read_table(self.cvPaths[model])
        y_real = []
        y_predproba = []
        ax = plt.gca()
        for fold in df.Fold.unique():
            # get labels
            y_pred = df[f"{property_name}_ProbabilityClass_1"][df.Fold == fold]
            y_true = df[f"{property_name}_Label"][df.Fold == fold]
            y_predproba.append(y_pred)
            y_real.append(y_true)
            # do plotting
            PrecisionRecallDisplay.from_predictions(
                y_true,
                y_pred,
                name="PRC fold {}".format(fold + 1),
                ax=ax,
                alpha=0.3,
                lw=1,
            )
        # Linear interpolation of PR curve is not recommended, so we don't plot "chance"
        # https://dl.acm.org/doi/10.1145/1143844.1143874
        # Plotting the average precision-recall curve over the cross validation runs
        y_real = np.concatenate(y_real)
        y_predproba = np.concatenate(y_predproba)
        PrecisionRecallDisplay.from_predictions(
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

    def makeInd(self, model: QSPRModel, property_name: str) -> plt.Axes:
        """Make the plot for a given model using independent test data.

        Args:
            model (QSPRModel):
                the model to plot the data from.
            property_name (str):
                name of the property to plot (should correspond to the prefix
                of the column names in the data files).

        Returns:
              ax (matplotlib.axes.Axes):
                the axes object containing the plot.
        """
        # read data from file
        df = pd.read_table(self.indPaths[model])
        y_pred = df[f"{property_name}_ProbabilityClass_1"]
        y_true = df[f"{property_name}_Label"]
        # do plotting
        ax = plt.gca()
        PrecisionRecallDisplay.from_predictions(
            y_true,
            y_pred,
            name="PRC",
            ax=ax,
        )
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=f"Receiver Operating Characteristic ({self.modelNames[model]})",
        )
        ax.legend(loc="best")
        return ax

    def make(
        self,
        save: bool = True,
        show: bool = False,
        property_name: str | None = None,
        validation: str = "cv",
        fig_size: tuple = (6, 6),
    ):
        """Make the plot for a given validation type.

        Args:
            property_name (str):
                name of the property to plot (should correspond to the prefix
                of the column names in the data files). If `None`, the first
                property in the model's `targetProperties` list will be used.
            validation (str):
                The type of validation data to use.
                Can be either 'cv' for cross-validation or 'ind'
                for independent test set.
            fig_size (tuple):
                The size of the figure to create.
            save (bool):
                Whether to save the plot to a file.
            show (bool):
                Whether to display the plot.

        Returns:
            axes (list): A list of matplotlib axes objects containing the plots.
        """
        if property_name is None:
            property_name = self.models[0].targetProperties[0].name
        choices = {"cv": self.makeCV, "ind": self.makeInd}
        axes = []
        for model in self.models:
            fig, ax = plt.subplots(figsize=fig_size)
            ax = choices[validation](model, property_name)
            axes.append(ax)
            if save:
                fig.savefig(f"{self.modelOuts[model]}.{validation}.png")
            if show:
                plt.show()
                plt.clf()
        return axes


class CalibrationPlot(ClassifierPlot):
    """Plot of calibration curve for a given model."""

    def getSupportedTasks(self) -> List[ModelTasks]:
        """Return a list of tasks supported by this plotter."""
        return [ModelTasks.SINGLECLASS, ModelTasks.MULTITASK_SINGLECLASS]

    def makeCV(
        self, model: QSPRModel, property_name: str, n_bins: int = 10
    ) -> plt.Axes:
        """Make the plot for a given model using cross-validation data.

        Args:
            model (QSPRModel):
                the model to plot the data from.
            property_name (str):
                name of the property to plot (should correspond to the
                prefix of the column names in the data files).
            n_bins (int):
                The number of bins to use for the calibration curve.

        Returns:
            ax (matplotlib.axes.Axes): the axes object containing the plot.
        """
        # read data from file for each fold and plot
        df = pd.read_table(self.cvPaths[model])
        y_real = []
        y_pred_proba = []
        ax = plt.gca()
        for fold in df.Fold.unique():
            # get labels
            y_pred = df[f"{property_name}_ProbabilityClass_1"][df.Fold == fold]
            y_true = df[f"{property_name}_Label"][df.Fold == fold]
            y_pred_proba.append(y_pred)
            y_real.append(y_true)
            # do plotting
            CalibrationDisplay.from_predictions(
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
        y_pred_proba = np.concatenate(y_pred_proba)
        CalibrationDisplay.from_predictions(
            y_real,
            y_pred_proba,
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

    def makeInd(
        self, model: QSPRModel, property_name: str, n_bins: int = 10
    ) -> plt.Axes:
        """Make the plot for a given model using independent test data.

        Args:
            model (QSPRModel):
                the model to plot the data from.
            property_name (str):
                name of the property to plot (should correspond to the prefix
                of the column names in the data files).
            n_bins (int):
                The number of bins to use for the calibration curve.

        Returns:
            ax (matplotlib.axes.Axes):
                the axes object containing the plot.
        """
        df = pd.read_table(self.indPaths[model])
        y_pred = df[f"{property_name}_ProbabilityClass_1"]
        y_true = df[f"{property_name}_Label"]
        ax = plt.gca()
        CalibrationDisplay.from_predictions(
            y_true,
            y_pred,
            n_bins=n_bins,
            name="Calibration",
            ax=ax,
        )
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=f"Calibration Curve ({self.modelNames[model]})",
        )
        ax.legend(loc="best")
        return ax

    def make(
        self,
        save: bool = True,
        show: bool = False,
        property_name: str | None = None,
        validation: str = "cv",
        fig_size: tuple = (6, 6),
    ) -> list[plt.Axes]:
        """Make the plot for a given validation type.

        Args:
            property_name (str):
                name of the property to plot (should correspond to the prefix
                of the column names in the data files). If `None`, the first
                property in the model's `targetProperties` list will be used.
            validation (str):
                The type of validation data to use. Can be either 'cv'
                for cross-validation or 'ind' for independent test set.
            fig_size (tuple):
                The size of the figure to create.
            save (bool):
                Whether to save the plot to a file.
            show (bool):
                Whether to display the plot.

        Returns:
            axes (list[plt.Axes]):
                A list of matplotlib axes objects containing the plots.
        """
        if property_name is None:
            property_name = self.models[0].targetProperties[0].name
        choices = {"cv": self.makeCV, "ind": self.makeInd}
        axes = []
        for model in self.models:
            fig, ax = plt.subplots(figsize=fig_size)
            ax = choices[validation](model, property_name, fig_size)
            axes.append(ax)
            if save:
                fig.savefig(f"{self.modelOuts[model]}.{validation}.png")
            if show:
                plt.show()
                plt.clf()
        return axes


class MetricsPlot(ClassifierPlot):
    """Plot of metrics for a given model.

    Attributes:
        models (list): A list of QSPRModel objects to plot the data from.
        metrics (list): A list of metrics to plot, choose from:
            f1, matthews_corrcoef, precision, recall, accuracy, roc_auc, roc_auc_ovr,
            roc_auc_ovo and calibration_error
    """

    def __init__(
        self,
        models: List[QSPRModel],
        metrics: List[
            Literal[
                "f1",
                "matthews_corrcoef",
                "precision",
                "recall",
                "accuracy",
                "calibration_error",
                "roc_auc",
                "roc_auc_ovr",
                "roc_auc_ovo",
            ]
        ] = [
            "f1",
            "matthews_corrcoef",
            "precision",
            "recall",
            "accuracy",
            "calibration_error",
            "roc_auc",
            "roc_auc_ovr",
            "roc_auc_ovo",
        ],
    ):
        """Initialise the metrics plot.

        Args:
            models (list): A list of QSPRModel objects to plot the data from.
            metrics (list): A list of metrics to plot.
        """
        super().__init__(models)
        self.metrics = metrics

    def make(
        self,
        save: bool = True,
        show: bool = False,
        out_path: str | None = None,
    ) -> tuple[List[sns.FacetGrid], pd.DataFrame]:
        """Make the plot for a given validation type.

        Args:
            property_name (str):
                name of the property to plot (should correspond to the prefix of
                the column names in the data files).
            save (bool):
                Whether to save the plot to a file.
            show (bool):
                Whether to display the plot.
            out_path (str | None):
                Path to save the plots to, e.g. "results/plot.png", the plot will be
                saved to this path with the metric name appended before the extension,
                e.g. "results/plot_roc_auc.png". If `None`, the plots will be saved to
                each model's output directory.

        Returns:
            figures (list[sns.FacetGrid]):
                the seaborn FacetGrid objects used to make the plot
            pd.DataFrame:
                A dataframe containing the summary data generated.
        """
        # Get summary with calculated metrics
        if not hasattr(self, "summary"):
            self.getSummary()

        figures = []
        for metric in self.metrics:
            # check if metric in summary dataframe
            if metric not in self.summary.columns:
                print(f"Metric {metric} not in summary dataframe, skipping")
                continue

            # plot the results
            g = sns.catplot(
                self.summary,
                x="Class",
                y=metric,
                hue="Set",
                col="Property",
                row="Model",
                kind="bar",
                margin_titles=True,
                sharex=False,
                sharey=False,
            )
            # set y range max to 1 for each plot, but don't set min to 0 as some metrics can be negative
            for ax in g.axes_dict.values():
                y_min, y_max = ax.get_ylim()
                ax.set_ylim(y_min, 1)
            figures.append(g)
            if save:
                # add metric to out_path if it exists before the extension
                if out_path is not None:
                    plt.savefig(
                        os.path.splitext(out_path)[0] + f"_{metric}.png", dpi=300
                    )
                else:
                    for model in self.models:
                        plt.savefig(f"{model.outPrefix}_{metric}.png", dpi=300)
            if show:
                plt.show()
                plt.clf()
        return figures, self.summary


class ConfusionMatrixPlot(ClassifierPlot):
    """Plot of confusion matrix for a given model as a heatmap."""

    def getConfusionMatrixDict(self, df: pd.DataFrame) -> dict:
        """Create dictionary of confusion matrices for each model, property and fold

        Args:
            df (pd.DataFrame):
                the dataframe containing the classficiation results,
                columns: Model, QSPRID, Fold, Property, Label, Prediction, Set

        Returns:
            dict:
                dictionary of confusion matrices for each model, property and fold
        """
        conf_dict = {}
        for model in df.Model.unique():
            for property in df.Property.unique():
                for fold in df.Fold.unique():
                    df_subset = df[
                        (df.Model == model)
                        & (df.Property == property)
                        & (df.Fold == fold)
                    ]
                    conf_dict[(model, property, fold)] = confusion_matrix(
                        df_subset.Label, df_subset.Prediction
                    )
        return conf_dict

    def make(
        self,
        save: bool = True,
        show: bool = False,
        out_path: str | None = None,
    ) -> tuple[dict, plt.Axes]:
        """Make confusion matrix heatmap for each model, property and fold

        Args:
            save (bool):
                whether to save the plot
            show (bool):
                whether to show the plot
            out_path (str | None):
                path to save the plot to, e.g. "results/plot.png", the plots will be
                saved to this path with the plot identifier appended before the extension,
                If `None`, the plots will be saved to each model's output directory.

        Returns:
            dict:
                dictionary of confusion matrices for each model, property and fold
            list[plt.axes.Axes]:
                a list of matplotlib axes objects containing the plots.
        """
        df = self.prepareClassificationResults()

        # Get dictionary of confusion matrices
        conf_dict = self.getConfusionMatrixDict(df)

        # Create heatmap for each model, property and fold
        axes = []
        for model in df.Model.unique():
            for property in df.Property.unique():
                for fold in df.Fold.unique():
                    fig, ax = plt.subplots()
                    sns.heatmap(
                        conf_dict[(model, property, fold)],
                        annot=True,
                        fmt="g",
                        cmap="Blues",
                    )
                    ax.set_title(f"Confusion Matrix ({model}_{property}_fold_{fold})")
                    ax.set_xlabel("Predicted label")
                    ax.set_ylabel("True label")
                    axes.append(fig)
                    if save:
                        if out_path is not None:
                            # add identifier to out_path before the extension
                            plt.savefig(
                                os.path.splitext(out_path)[0]
                                + f"_{model}_{property}_{fold}_confusion_matrix.png",
                                dpi=300,
                            )
                        else:
                            # reverse self.modelNames dictionary to get model out
                            modelNames = {v: k for k, v in self.modelNames.items()}
                            plt.savefig(
                                f"{self.modelOuts[modelNames[model]]}_{property}_{fold}_confusion_matrix.png",
                                dpi=300,
                            )
                    if show:
                        plt.show()
                        plt.clf()
                    plt.close()
        return axes, conf_dict
