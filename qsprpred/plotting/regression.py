"""Module for plotting regression models."""
from abc import ABC
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics

from ..data import QSPRDataset
from ..models import QSPRModel
from ..plotting.base_plot import ModelPlot
from ..tasks import ModelTasks


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
            )
            .reset_index(level=1, drop=True)
            .reset_index()
        )

        self.results = df
        return df

    def getSummary(self):
        """calculate the R2 and RMSE for each model per set (cross-validation or independent test)"""
        if not hasattr(self, "results"):
            self.prepareRegressionResults()
        df = deepcopy(self.results)
        df_summary = (
            df.groupby(["Model", "Fold", "Property"])
            .apply(
                lambda x: pd.Series(
                    {
                        "R2": metrics.r2_score(x["Label"], x["Prediction"]),
                        "RMSE": metrics.root_mean_squared_error(
                            x["Label"], x["Prediction"]
                        ),
                    }
                )
            )
            .reset_index()
        )
        df_summary["Set"] = df_summary["Fold"].apply(
            lambda x: "Independent Test"
            if x == "Independent Test"
            else "Cross Validation"
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

    def __init__(self, models: list[QSPRModel], datasets: list[QSPRDataset]):
        super().__init__(models)
        self.datasets = datasets

    def make(
        self,
        save: bool = True,
        show: bool = False,
        out_path: str | None = None,
    ) -> tuple[sns.FacetGrid, pd.DataFrame, List[float]]:
        """make Williams plot

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
                the leverages and standardized residuals for each compound
            dict[str, float]:
                the h* values for the datasets
        """

        def calculateLeverages(
            features_train: pd.DataFrame, features_test: pd.DataFrame
        ) -> pd.DataFrame:
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
            h_star = (3 * (p + 1)) / N

            # print waring if h* > 1
            if h_star > 1:
                print(
                    f"Warning: h* = {h_star} is greater than 1, this may indicate that the "
                    "number of samples is too small for the number of features. "
                    "Leverage values are between 0 and 1, so h* should be less than 1."
                )

            return leverages, h_star

        # prepare the dataframe for plotting
        df = self.prepareRegressionResults()

        # calculate the leverages and h* for each model
        model_leverages = {}
        model_h_star = {}
        model_p = {}  # number of descriptors
        for model, dataset in zip(self.models, self.datasets):
            model_name = model.name
            if dataset.hasFeatures:
                features = dataset.getFeatures()
                leverages, h_star = calculateLeverages(*features)
                model_leverages[model_name] = leverages
                model_h_star[model_name] = h_star
                model_p[model_name] = features[0].shape[1]
            else:
                raise ValueError(
                    f"Dataset {dataset.name} does not have features, to"
                    " calculate leverages, the dataset should have features."
                )

        # Add the levarages to the dataframe
        df["leverage"] = df.apply(
            lambda x: model_leverages[x["Model"]][x["QSPRID"]], axis=1
        )
        df["n_features"] = df["Model"].apply(lambda x: model_p[x])

        # calculate the residuals
        df["residual"] = df["Label"] - df["Prediction"]

        # calculate the residuals standard deviation
        df["n_samples"] = df.groupby(["Model", "Set", "Property"])[
            "residual"
        ].transform("count")

        # calculate degrees of freedom
        df["df"] = df["n_samples"] - df["n_features"] - 1

        RSE = {}
        # check if the degrees of freedom is greater than 0 for each model, property, and set
        for (model, set, property), df_ in df.groupby(["Model", "Set", "Property"]):
            if set == "Cross Validation":
                if df_["df"].iloc[0] <= 0:
                    print(f"{model} {set} {property}")
                    print(df_[["n_samples", "n_features"]].iloc[0])
                    raise ValueError(
                        "Degrees of freedom is less than or equal to 0 for some models, "
                        "properties trainingset. Check the number of samples and features, the "
                        "number of samples should be greater than the number of features."
                    )
                RSE[(model, property)] = np.sqrt(
                    (1 / df_["df"].iloc[0]) * np.sum(df_["residual"] ** 2)
                )

        # add the residual standard error to the df
        df["RSE"] = df.apply(lambda x: RSE[(x["Model"], x["Property"])], axis=1)

        # calculate the standardized residuals
        df["std_resid"] = df["residual"] / (df["RSE"] * np.sqrt(1 - df["leverage"]))

        # plot the results
        g = sns.FacetGrid(
            df,
            col="Property",
            row="Model",
            margin_titles=True,
            height=4,
            sharex=False,
            sharey=False,
            hue="Set",
        )
        g.map(sns.scatterplot, "leverage", "std_resid", s=7, edgecolor="none")
        # add the h* line to each plot based on the model's h*
        # and add hlines at +/- 3
        for k, ax in g.axes_dict.items():
            ax.axvline(model_h_star[k[0]], c=".2", ls="--")
            ax.axhline(2, c=".2", ls="--")
            ax.axhline(-2, c=".2", ls="--")

        # set y axis tile to standardized residuals
        g.set_ylabels("Studentized Residuals")
        g.add_legend()

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
        return (
            g,
            df[["Model", "Fold", "Property", "leverage", "std_resid", "QSPRID"]],
            model_h_star,
        )
