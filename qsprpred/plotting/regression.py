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
        self,
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
    """Class to plot the results of regression models. Plot predicted pX_train vs real pX_train."""
    def make(
        self,
        save: bool = True,
        show: bool = False,
        out_path: str | None = None,
    ) -> tuple[sns.FacetGrid, pd.DataFrame]:
        """Plot the results of regression models. Plot predicted pX_train vs real pX_train.

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
        df = self.prepareRegressionResults()

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

        # save the plot
        if save:
            if out_path is not None:
                plt.savefig(out_path, dpi=300)
            else:
                for model in self.models:
                    plt.savefig(f"{model.outPrefix}_correlation.png", dpi=300)
        # show the plot
        if show:
            plt.show()

        plt.clf()
        return g, self.summary

class WilliamsPlot(RegressionPlot):
    """Williams plot; plot of standardized residuals versus leverages"""
    def make(
        self,
        datasets: dict[str, QSPRDataset] | None = None,
        save: bool = True,
        show: bool = False,
        out_path: str | None = None,
    ) -> tuple[sns.FacetGrid, pd.DataFrame]:
        """make Williams plot

        Args:
            datasets (dict[str, QSPRDataset] | None):
                dictionary of datasets to use for the plot, keys are the model names.
                If None, the models should have datasets attached to them.
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
        def calculateLeverages(features_train: pd.DataFrame, features_test: pd.DataFrame) -> pd.DataFrame:
            """Calculate the leverages for each compound in the dataset.

            Args:
                features_train (pd.DataFrame):
                    the features for each compound in the training set
                features_test (pd.DataFrame):
                    the features for each compound in the test set

            Returns:
                pd.DataFrame:
                    the leverages for each compound in the dataset
                float:
                    the h* value for the dataset
            """
            X_train = features_train.values
            X_test = features_test.values

            # assert the number of samples is greater than the number of features
            assert X_train.shape[0] > X_train.shape[1], (
                f"The number of samples ({X_train.shape[0]}) should be greater than the "
                f"number of features ({X_train.shape[1]}) for calculating the leverages."
            )

            # get the diagonal elements of the hat matrix
            # these are the leverages
            pinv_XTX = np.linalg.pinv(X_train.T @ X_train)
            leverages_train = np.diag(X_train @ pinv_XTX @ X_train.T)
            leverages_train = pd.Series(leverages_train, index=features_train.index)

            leverages_test = np.diag(X_test @ pinv_XTX @ X_test.T)
            leverages_test = pd.Series(leverages_test, index=features_test.index)

            leverages = pd.concat([leverages_train, leverages_test])

            # h* = (3(p+1)/N) is the cutoff for high leverage points
            # p is the number of features
            # N is the number of compounds
            p = X_train.shape[1]
            N = X_train.shape[0]
            h_star = (3*(p+1))/N

            return leverages, h_star

        # prepare the dataframe for plotting
        df = self.prepareRegressionResults()

        if datasets is None:
            datasets = {}
            for model in self.models:
                if model.checkForDataset():
                    datasets[model.name] = model.dataset
                else:
                    raise ValueError(
                        "Model does not have a dataset attached to it."
                    )

        # calculate the leverages and h* for each model
        model_leverages = {}
        model_h_star = {}
        model_p = {} # number of descriptors
        for model_name, dataset in datasets.items():
            if dataset.hasFeatures:
                features = dataset.getFeatures()
                leverages, h_star = calculateLeverages(*features)
                model_leverages[model_name] = leverages
                model_h_star[model_name] = h_star
                model_p[model_name] = features[0].shape[1]
            else:
                raise ValueError(
                    f"Dataset {dataset.name} does not have features, to"
                    " calculate leverages, the dataset should have features ."
                )

        # Add the levarages to the dataframe
        df["leverage"] = df.apply(lambda x: model_leverages[x["Model"]][x["QSPRID"]], axis=1)
        df["n_features"] = df["Model"].apply(lambda x: model_p[x])

        # calculate the residuals
        df["residual"] = df["Label"] - df["Prediction"]

        # calculate the residuals standard deviation
        df["n_samples"]  = df.groupby(["Model", "Fold", "Property"])["residual"].transform("count")

        # calculate degrees of freedom
        df["df"] = df["n_samples"] - df["n_features"] - 1

        # check if the degrees of freedom is greater than 0 for each model, property, and fold
        if (df["df"] <= 0).any():
            for (model, fold, property), df_ in df.groupby(["Model", "Fold", "Property"]):
                if df_["df"].iloc[0] <= 0:
                    print(f"{model} {fold} {property}")
                    print(df_[["n_samples", "n_features"]].iloc[0])
            raise ValueError(
                "Degrees of freedom is less than or equal to 0 for some models, "
                "properties, and folds. Check the number of samples and features, the "
                "number of samples should be greater than the number of features."
            )

        # calculate the residual standard error
        df["RSE"] = np.sqrt(
            (1/df["df"])*np.sum(df["residual"]**2)
        )

        # calculate the standardized residuals
        df["std_resid"] = df["residual"] / (df["RSE"]*np.sqrt(1-df["leverage"]))

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
            ax.axhline(2, c=".2", ls="--")
            ax.axhline(-2, c=".2", ls="--")

        # save the plot
        if save:
            if out_path is not None:
                plt.savefig(out_path, dpi=300)
            else:
                for model in self.models:
                    plt.savefig(f"{model.outPrefix}_williamsplot.png", dpi=300)
        # show the plot
        if show:
            plt.show()
        plt.clf()
        return g
