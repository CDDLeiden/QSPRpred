"""Module for plotting regression models."""

from abc import ABC
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics

from ..data.tables.interfaces.qspr_data_set import QSPRDataSet
from ..models import QSPRModel
from ..plotting.base_plot import ModelPlot
from ..tasks import ModelTasks


class RegressionPlot(ModelPlot, ABC):
    """Base class for all regression plots."""

    def getSupportedTasks(self) -> list[ModelTasks]:
        """Return a list of supported model tasks."""
        return [ModelTasks.REGRESSION, ModelTasks.MULTITASK_REGRESSION]

    def prepareAssessment(self, name: str, assessment_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare assessment dataframe for plotting

        Args:
            name (str):
                the name of the assessment
            assessment_df (pd.DataFrame):
                the assessment dataframe containing the experimental and predicted
                values for each property. The dataframe should have the following
                columns:
                ID, Fold , <property_name>_<suffixes>_<Label/Prediction>

        Returns:
            pd.DataFrame:
                The dataframe containing the assessment results,
                columns: ID, Fold, Set, Property, Label, Prediction, Set
        """
        # Melt all property columns into one column
        id_vars = ["ID", "Fold", "Set"]
        df = assessment_df.melt(id_vars=id_vars)
        # split the variable (<property_name>_<suffixes>_<Label/Prediction>) column
        # into the property name and the type (Label or Prediction)
        df[["Property", "type"]] = df["variable"].str.rsplit("_", n=1, expand=True)
        df.drop("variable", axis=1, inplace=True)
        # pivot the dataframe so that Label and Prediction are separate columns
        df = df.pivot_table(
            index=[*id_vars, "Property"], columns="type", values="value"
        )
        df.dropna(subset=["Label"], inplace=True) # Remove rows with missing values in the label column
        df.reset_index(inplace=True)
        df.columns.name = None
        df["Assessment"] = name

        return df

    def prepareRegressionResults(self) -> pd.DataFrame:
        """Prepare regression results dataframe for plotting.

        Returns:
            pd.DataFrame:
                the dataframe containing the regression results,
                columns: Model, QSPRID, Fold, Property, Label, Prediction, Set
        """
        model_results = {}
        for model in self.models:
            # Read in and prepare the assessment set results
            results = []
            for name, path in self.assesmentPaths[model].items():
                assessment = pd.read_table(path)
                # Replace the assessment labels with the dataset labels
                if self.datasets is not None and model.name in self.datasets:
                    targets = self.datasets[model.name].getTargets()
                    assessment = assessment.drop(columns=[col for col in assessment.columns if col.endswith("Label")])
                    # add suffix Label to the target columns
                    targets.columns = [f"{col}_Label" for col in targets.columns]
                    assessment = pd.merge(assessment, targets, on="ID")
                df = self.prepareAssessment(name, assessment)
                results.append(df)
            df = pd.concat(results)
            print(model.name)
            model_results[model.name] = df
        # concatenate the results from all models and add the model name as a column
        df = pd.concat(model_results)

        self.results = df
        return df

    def getSummary(self) -> pd.DataFrame:
        """calculate the R2 and RMSE for each model per set (cross-validation or independent test)"""
        if not hasattr(self, "results"):
            self.prepareRegressionResults()
        df = deepcopy(self.results)
        df_summary = (
            df.groupby(["Model", "Assessment", "Fold", "Set", "Property"]).apply(
                lambda x: pd.Series(
                    {
                        "R2":
                            metrics.r2_score(x["Label"], x["Prediction"]),
                        "RMSE":
                            metrics.
                            root_mean_squared_error(x["Label"], x["Prediction"]),
                    }
                )
            ).reset_index()
        )
        self.summary = df_summary
        return self.summary


class CorrelationPlot(RegressionPlot):
    """Class to plot the results of regression models. Plot predicted pX_train vs real pX_train."""

    def make(
        self,
        save: bool = True,
        show: bool = False,
        out_path: str | None = None
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

        # Select only test set results
        df = df[df["Set"] == "Test"]

        # plot the results
        g = sns.FacetGrid(
            df,
            col="Property",
            row="Model",
            hue="Assessment",
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
    def __init__(self, models: list[QSPRModel], assessments: list[str], datasets: list[QSPRDataSet]):
        """Initialize the Williams plot.
        Args:
            models (list[QSPRModel]):
                the models to plot
            assessments (list[str]):
                the assessments to plot
            datasets (list[QSPRDataSet]):
                the datasets to plot
        """
        super().__init__(models, assessments, datasets)

    def make(
        self,
        property_name: str,
        save: bool = True,
        show: bool = False,
        out_path: str | None = None
    ) -> tuple[sns.FacetGrid, pd.DataFrame, List[float]]:
        """make Williams plot

        Args:
            property_name (str):
                the name of the property to plot
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
            X_train = features_train.values.astype(float)
            X_test = features_test.values.astype(float)

            # assert the number of samples is greater than the number of features
            assert X_train.shape[0] > X_train.shape[1], (
                f"The number of samples ({X_train.shape[0]}) should be greater than the "
                f"number of features ({X_train.shape[1]}) for calculating the leverages."
            )

            # get the diagonal elements of the hat matrix
            # these are the leverages
            pinv_XTX = np.linalg.pinv(X_train.T @ X_train)
            leverages_train = np.sum(X_train @ pinv_XTX * X_train, axis=1)
            leverages_train = pd.Series(leverages_train, index=features_train.index)

            leverages_test = np.sum(X_test @ pinv_XTX * X_test, axis=1)
            leverages_test = pd.Series(leverages_test, index=features_test.index)

            leverages = pd.concat([leverages_train, leverages_test])

            # h* = (3(p+1)/N) is the cutoff for high leverage points
            # p is the number of features
            # N is the number of compounds
            p = X_train.shape[1]
            N = X_train.shape[0]
            h_star = (3 * (p + 1)) / N

            if h_star > 1:
                print(
                    f"Warning: h* = {h_star} is greater than 1, this may indicate that the "
                    "number of samples is too small for the number of features. "
                    "Leverage values are between 0 and 1, so h* should be less than 1."
                )

            return leverages, h_star

        # prepare the dataframe for plotting
        df = self.prepareRegressionResults()

        # Select property to plot
        df = df[df["Property"] == property_name].copy()

        # calculate the leverages and h* for each model, assessment and fold
        model_leverages = {}
        model_h_star = {}
        model_p = {}  # number of descriptors
        for model in self.models:
            dataset = self.datasets[model.name]
            for assessment in df["Assessment"].unique():
                df_assessment = df[(df["Model"] == model.name) & (df["Assessment"] == assessment)]
                for fold in df_assessment["Fold"].unique():
                    df_ = df_assessment[(df_assessment["Fold"] == fold)]
                    train_ind = df_[df_["Set"] == "Train"]["ID"].to_list()
                    test_ind = df_[df_["Set"] == "Test"]["ID"].to_list()
                    # FIXME: Pipeline is refit for each fold, this is not ideal
                    # this information should be ideally be retrieved from the
                    # assessment, pipeline or similar
                    pipeline = deepcopy(model.pipeline)
                    X_train, _ = next(pipeline.apply(dataset[train_ind], seed=model.randomState))
                    X_test, _ = next(pipeline.apply(dataset[test_ind], fit=False, seed=model.randomState))
                    leverages, h_star = calculateLeverages(X_train, X_test)

                    model_name = model.name
                    model_leverages[f"{model_name}_{assessment}_{fold}"] = leverages
                    model_h_star[f"{model_name}_{assessment}_{fold}"] = h_star
                    model_p[f"{model_name}_{assessment}_{fold}"] = X_train.shape[1]

        # Add the leverages to the dataframe
        df["leverage"] = df.apply(
            lambda x: model_leverages[f"{x['Model']}_{x['Assessment']}_{x['Fold']}"][x["ID"]],
            axis=1,
        )
        df["n_features"] = df.apply(
            lambda x: model_p[f"{x['Model']}_{x['Assessment']}_{x['Fold']}"], axis=1
        )

        # calculate the residuals
        df["residual"] = df["Label"] - df["Prediction"]

        # calculate the residuals standard deviation
        df["n_samples"] = df.groupby(
            ["Model", "Assessment", "Property", "Fold", "Set"]
        )["residual"].transform("count")

        # calculate degrees of freedom
        df["df"] = df["n_samples"] - df["n_features"] - 1

        RSE = {}
        # check if the degrees of freedom is greater than 0 for each model, assessment, property, and set
        for (model, property, assessment, fold, set), df_ in df.groupby(["Model", "Property", "Assessment", "Fold", "Set"]):
            if df_["df"].iloc[0] <= 0:
                raise ValueError(
                    "Degrees of freedom is less than or equal to 0 for some models, "
                    "properties trainingset. Check the number of samples and features, the "
                    "number of samples should be greater than the number of features."
                )
            RSE[(model, property, assessment, fold, set)] = np.sqrt(
                (1 / df_["df"].iloc[0]) * np.sum(df_["residual"]**2)
            )

        # add the residual standard error to the df
        df["RSE"] = df.apply(lambda x: RSE[(x["Model"], x["Property"], x["Assessment"], x["Fold"], x["Set"])], axis=1)

        # calculate the standardized residuals
        df["std_resid"] = df["residual"] / (df["RSE"] * np.sqrt(1 - df["leverage"]))

        # plot the results
        g = sns.FacetGrid(
            df[df["Set"] == "Test"],
            col="Property",
            row="Model",
            margin_titles=True,
            height=4,
            sharex=False,
            sharey=False,
            hue="Assessment",
        )
        g.map(sns.scatterplot, "leverage", "std_resid", s=7, edgecolor="none")
        # add the h* line to each plot based on the model's h*
        # and add hlines at +/- 3
        for k, ax in g.axes_dict.items():
            for assessment in df[df["Model"] == k[0]]["Assessment"].unique():
                for fold in df[(df["Model"] == k[0]) & (df["Assessment"] == assessment)]["Fold"].unique():
                    ax.axvline(model_h_star[f"{k[0]}_{assessment}_{fold}"], c=".2", ls="--")
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
            df[["ID", "Model", "Property", "Assessment", "Fold", "Set", "leverage", "std_resid"]],
            model_h_star,
        )
