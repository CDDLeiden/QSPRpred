"""Module for plotting regression models."""
import math
from abc import ABC

import pandas as pd
from matplotlib import pyplot as plt
from qsprpred.models.tasks import ModelTasks
from qsprpred.plotting.interfaces import ModelPlot
from sklearn import metrics


class RegressionPlot(ModelPlot, ABC):
    """Base class for all regression plots."""

    def getSupportedTasks(self):
        """Return a list of supported model tasks."""
        return [ModelTasks.REGRESSION]


class CorrelationPlot(RegressionPlot):
    """Class to plot the results of regression models. Plot predicted pX vs real pX."""

    def make(self, property_name: str, save: bool = True, show: bool = False,
             out_dir: str = ".", filename_prefix: str = "corrplot"):
        """Plot the results of regression models. Plot predicted pX vs real pX.

        Args:
            property_name (`str`): name of the property to plot (should correspond to the prefix of the column names in the data files)
            save (`bool`): whether to save the plot to a file
            show (`bool`): whether to show the plot
            out_dir (`str`): directory to save the plot to
            filename_prefix (`str`): prefix to use for the filename

        Returns:
            ret_axes (`matplotlib.axes.Axes`): the axes of the plot
            summary (`pandas.DataFrame`): a summary of the plot
        """
        my_cmap = ["#12517B", "#88002A"]

        plt.figure(figsize=(5, 5))
        cate = [self.cvPaths, self.indPaths]
        cate_names = ["cv", "ind"]
        ret_axes = []
        summary = {"ModelName": [], "R2": [], "RMSE": [], "Set": []}
        for m, model in enumerate(self.models):
            ax = plt.subplot(1, len(self.models), m + 1)
            ret_axes.append(ax)
            min_val = 0
            max_val = 10
            for j, legend in enumerate(['Cross Validation', 'Independent Test']):
                df = pd.read_table(cate[j][model])
                plt.scatter(
                    df[f"{property_name}_Label"],
                    df[f"{property_name}_Prediction"],
                    s=5,
                    label=legend,
                    color=my_cmap[j])
                coef = metrics.r2_score(df[f"{property_name}_Label"], df[f"{property_name}_Prediction"])
                rmse = metrics.mean_squared_error(
                    df[f"{property_name}_Label"],
                    df[f"{property_name}_Prediction"],
                    squared=False)
                summary["R2"].append(coef)
                summary["RMSE"].append(rmse)
                summary["Set"].append(cate_names[j])
                summary["ModelName"].append(model.name)

                plt.title(model)
                plt.xlabel(f"Experimental {property_name}")
                plt.ylabel(f"Predicted {property_name}")
                min_val_now = math.floor(
                    min(pd.concat([df[f"{property_name}_Label"], df[f"{property_name}_Prediction"]])))
                max_val_now = math.ceil(
                    max(pd.concat([df[f"{property_name}_Label"], df[f"{property_name}_Prediction"]])))
                if min_val_now < min_val:
                    min_val = min_val_now
                if max_val_now > max_val:
                    max_val = max_val_now
                pad = (max_val - min_val) * 0.1
                plt.plot(
                    [min_val - pad, max_val + pad],
                    [min_val - pad, max_val + pad],
                    lw=2, linestyle='--', color='black')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        if show:
            plt.show()
            plt.clf()

        if save:
            plt.savefig(f"{out_dir}/{filename_prefix}.png", dpi=300)

        return ret_axes, pd.DataFrame(summary)
