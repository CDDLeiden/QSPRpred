"""Module for plotting regression models."""
import math
from abc import ABC

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics

from ..models.tasks import ModelTasks
from ..plotting.interfaces import ModelPlot


class RegressionPlot(ModelPlot, ABC):
    """Base class for all regression plots."""
    def getSupportedTasks(self) -> list[ModelTasks]:
        """Return a list of supported model tasks."""
        return [ModelTasks.REGRESSION]


class CorrelationPlot(RegressionPlot):
    """Class to plot the results of regression models. Plot predicted pX vs real pX."""
    def make(
        self,
        save: bool = True,
        show: bool = False,
        property_name: str | None = None,
        out_dir: str = ".",
        filename_prefix: str = "corrplot",
        n_cols: int = 1,
    ) -> [list[plt.Axes], pd.DataFrame]:
        """Plot the results of regression models. Plot predicted pX vs real pX.

        Args:
            property_name (str):
                the name of the property to plot. If `None`, the first property in the
                targetProperties list of the model will be used.
            save (bool):
                whether to save the plot
            show (bool):
                whether to show the plot
            out_dir (str):
                the directory to save the plot
            filename_prefix (str):
                the prefix of the filename
            n_cols (int):
                the number of columns in the plot

        Returns:
            list[plt.Axes]:
                the list of axes objects used to make the plot
            pd.DataFrame:
                the summary data used to make the plot
        """
        if property_name is None:
            property_name = self.models[0].targetProperties[0].name
        my_cmap = ["#12517B", "#88002A"]
        ln = len(self.models)
        n_rows = math.ceil(ln / n_cols)
        plt.figure(figsize=(5 * n_cols, 5 * n_rows))
        cate = [self.cvPaths, self.indPaths]
        cate_names = ["cv", "ind"]
        ret_axes = []
        summary = {"ModelName": [], "R2": [], "RMSE": [], "Set": []}
        for m, model in enumerate(self.models):
            ax = plt.subplot(n_rows, n_cols, m + 1)
            ret_axes.append(ax)
            min_val = np.inf
            max_val = -np.inf
            for j, legend in enumerate(["Cross Validation", "Independent Test"]):
                df = pd.read_table(cate[j][model])
                plt.scatter(
                    df[f"{property_name}_Label"],
                    df[f"{property_name}_Prediction"],
                    s=5,
                    label=legend,
                    color=my_cmap[j],
                )
                coef = metrics.r2_score(
                    df[f"{property_name}_Label"], df[f"{property_name}_Prediction"]
                )
                rmse = metrics.mean_squared_error(
                    df[f"{property_name}_Label"],
                    df[f"{property_name}_Prediction"],
                    squared=False,
                )
                summary["R2"].append(coef)
                summary["RMSE"].append(rmse)
                summary["Set"].append(cate_names[j])
                summary["ModelName"].append(model.name)
                # plot the line
                plt.title(model)
                plt.xlabel(f"Experimental {property_name}")
                plt.ylabel(f"Predicted {property_name}")
                min_val_now = math.floor(
                    min(
                        pd.concat(
                            [
                                df[f"{property_name}_Label"],
                                df[f"{property_name}_Prediction"],
                            ]
                        )
                    )
                )
                max_val_now = math.ceil(
                    max(
                        pd.concat(
                            [
                                df[f"{property_name}_Label"],
                                df[f"{property_name}_Prediction"],
                            ]
                        )
                    )
                )
                if min_val_now < min_val:
                    min_val = min_val_now
                if max_val_now > max_val:
                    max_val = max_val_now
                pad = (max_val - min_val) * 0.1
                plt.plot(
                    [min_val - pad, max_val + pad],
                    [min_val - pad, max_val + pad],
                    lw=2,
                    linestyle="--",
                    color="black",
                )
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        # show the plot
        if show:
            plt.show()
            plt.clf()
        # save the plot
        if save:
            plt.savefig(f"{out_dir}/{filename_prefix}.png", dpi=300)
        return ret_axes, pd.DataFrame(summary)
