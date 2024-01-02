"""Module for plotting regression models."""
from abc import ABC
from copy import deepcopy
from typing import List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics

from ..tasks import ModelTasks
from ..plotting.base_plot import ModelPlot
from ..data import QSPRDataset
import numpy as np


class RegressionPlot(ModelPlot, ABC):
    """Base class for all regression plots."""
    def getSupportedTasks(self) -> list[ModelTasks]:
        """Return a list of supported model tasks."""
        return [ModelTasks.REGRESSION, ModelTasks.MULTITASK_REGRESSION]

    def prepareAssessment(self, assessment_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare assessment dataframe for plotting

        Args:
            assessment_df (pd.DataFrame):
                the assessment dataframe containing the experimental and predicted
                values for each property. The dataframe should have the following
                columns:
                QSPRID, Fold (opt.), <property_name>_<suffixes>_<Label/Prediction>

        Returns:
            pd.DataFrame:
                The dataframe containing the assessment results,
                columns: QSPRID, Fold, Property, Label, Prediction, Set
        """
        # change all property columns into one column
        id_vars = ["QSPRID", "Fold"] if "Fold" in assessment_df.columns else ["QSPRID"]
        df = assessment_df.melt(id_vars=id_vars)
        # split the variable (<property_name>_<suffixes>_<Label/Prediction>) column
        # into the property name and the type (Label or Prediction)
        df[["Property", "type"]] = df["variable"].str.rsplit("_", n=1, expand=True)
        df.drop("variable", axis=1, inplace=True)
        # pivot the dataframe so that Label and Prediction are separate columns
        df = df.pivot_table(
            index=[*id_vars, "Property"], columns="type", values="value"
        )
        df.reset_index(inplace=True)
        df.columns.name = None
        # Add Fold column if it doesn't exist (for independent test set)
        if "Fold" not in df.columns:
            df["Fold"] = "Independent Test"
            df["Set"] = "Independent Test"
        else:
            df["Set"] = "Cross Validation"
        return df

    def prepareRegressionResults(
        self, property_name: str | None = None
    ) -> pd.DataFrame:
        """Prepare regression results dataframe for plotting.

        Returns:
            pd.DataFrame:
                the dataframe containing the regression results,
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
            ).reset_index(level=1, drop=True).reset_index()
        )

        self.results = df
        return df

    def getSummary(self):
        """calculate the R2 and RMSE for each model per set (cross-validation or independent test)"""
        if not hasattr(self, "results"):
            self.prepareRegressionResults()
        df = deepcopy(self.results)
        df_summary = (
            df.groupby(["Model", "Fold", "Property"]).apply(
                lambda x: pd.Series(
                    {
                        "R2":
                            metrics.r2_score(x["Label"], x["Prediction"]),
                        "RMSE":
                            metrics.mean_squared_error(
                                x["Label"], x["Prediction"], squared=True
                            ),
                    }
                )
            ).reset_index()
        )
        df_summary["Set"] = df_summary["Fold"].apply(
            lambda x: "Independent Test"
            if x == "Independent Test" else "Cross Validation"
        )
        self.summary = df_summary
        return df_summary


class CorrelationPlot(RegressionPlot):
    """Class to plot the results of regression models. Plot predicted pX vs real pX."""
    def make(
        self,
        save: bool = True,
        show: bool = False,
        property_name: str | None = None,
        out_path: str | None = None,
    ) -> tuple[sns.FacetGrid, pd.DataFrame]:
        """Plot the results of regression models. Plot predicted pX vs real pX.

        Args:
            save (bool):
                whether to save the plot
            show (bool):
                whether to show the plot
            out_path (str | None):
                path to save the plot to, e.g. "results/plot.png", if `None`, the plot
                will be saved to each model's output directory.

        Returns:
            g (sns.FacetGrid):
                the seaborn FacetGrid object used to make the plot
            pd.DataFrame:
                the summary data used to make the plot
        """
        # prepare the dataframe for plotting
        df = self.prepareRegressionResults(property_name)

        if not hasattr(self, "summary"):
            self.getSummary()

        # plot the results
        g = sns.FacetGrid(
            df,
            col="Property",
            row="Model",
            hue="Set",
            margin_titles=True,
            height=4,
            sharex=False,
            sharey=False,
        )
        g.map(sns.scatterplot, "Label", "Prediction", s=7, edgecolor="none")
        # set x and y range to be the same for each plot
        for ax in g.axes_dict.values():
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            ax_min = min(x_min, y_min)
            ax_max = max(x_max, y_max)
            pad = (ax_max - ax_min) * 0.1
            ax.set_xlim(ax_min - pad, ax_max + pad)
            ax.set_ylim(ax_min - pad, ax_max + pad)
            ax.set_aspect("equal", "box")
        for ax in g.axes_dict.values():
            ax.axline((0, 0), slope=1, c=".2", ls="--")

        g.add_legend()

        # show the plot
        if show:
            plt.show()
        # save the plot
        if save:
            if out_path is not None:
                plt.savefig(out_path, dpi=300)
            else:
                for model in self.models:
                    plt.savefig(f"{model.outPrefix}_correlation.png", dpi=300)
        plt.clf()
        return g, self.summary

class WilliamsPlot(RegressionPlot):
    """Williams plot: plot of standardized residuals ( y -axis) versus leverages (hat values; x axis)"""
    def make(
        self,
        datasets: List[QSPRDataset]|None = None,
        save: bool = True,
        show: bool = False,
        property_name: str | None = None,
        out_path: str | None = None,
    ) -> tuple[sns.FacetGrid, pd.DataFrame]:
        """make Williams plot

        Args:
            datasets (List[QSPRDataSet]|None):
                list of datasets to use for the plot. Should be a list of QSPRDataSet
                in the same order as the models. If None, the models should have
                datasets attached to them.
            save (bool):
                whether to save the plot
            show (bool):
                whether to show the plot
            out_path (str | None):
                path to save the plot to, e.g. "results/plot.png", if `None`, the plot
                will be saved to each model's output directory.

        Returns:
            g (sns.FacetGrid):
                the seaborn FacetGrid object used to make the plot
        """
        def calculateLeverages(descriptors: pd.DataFrame) -> pd.DataFrame:
            """Calculate the leverages for each compound in the dataset."""
            X = descriptors.values

            # get the diagonal elements of the hat matrix
            # these are the leverages
            leverages = np.diag(X @ np.linalg.pinv(X.T @ X) @ X.T)
            leverages = pd.Series(leverages, index=descriptors.index)

            # h* = (3(p+1)/N) is the cutoff for high leverage points
            # p is the number of descriptors
            # N is the number of compounds
            p = X.shape[1]
            N = X.shape[0]
            h_star = (3*(p+1))/N

            return leverages, h_star

        # prepare the dataframe for plotting
        df = self.prepareRegressionResults(property_name)

        if datasets is None:
            datasets = []
            for model in self.models:
                if model.checkForDataset():
                    datasets = [model.dataset]
                else:
                    raise ValueError(
                        "Model does not have a dataset attached to it."
                    )

        model_leverages = {}
        model_h_star = {}
        for i, dataset in enumerate(datasets):
            if dataset.hasDescriptors:
                descriptors = dataset.getDescriptors()
                leverages, h_star = calculateLeverages(descriptors)
                model_leverages[self.models[i].name] = leverages
                model_h_star[self.models[i].name] = h_star
            else:
                raise ValueError(
                    f"Dataset {dataset.name} does not have descriptors, to"
                    " calculate leverages, the dataset should have descriptors."
                )
        df["leverage"] = df.apply( lambda x: model_leverages[x["Model"]][x["QSPRID"]], axis=1)
        df["residual"] = df["Label"] - df["Prediction"]
        variance = df.groupby(["Model", "Fold", "Property"]).apply(
            lambda x: np.var(x["residual"])
        ).reset_index()
        df["std_resid"] = df.apply( lambda x: x["residual"] / 
                
        # plot the results
        g = sns.FacetGrid(
            df,
            col="Property",
            row="Model",
            margin_titles=True,
            height=4,
            sharex=False,
            sharey=False,
            hue="Set"
        )
        g.map(sns.scatterplot, "leverage", "std_resid", s=7, edgecolor="none")
        # add the h* line to each plot based on the model's h*
        # and add hlines at +/- 3
        for k, ax in g.axes_dict.items():
            ax.axvline(model_h_star[k[0]], c=".2", ls="--")
            ax.axhline(3, c=".2", ls="--")
            ax.axhline(-3, c=".2", ls="--")
        return g