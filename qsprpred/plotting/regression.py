"""
regression

Created by: Martin Sicho
On: 12.02.23, 18:15
"""
import math
from abc import ABC

import pandas as pd
from matplotlib import pyplot as plt
from qsprpred.models.tasks import TargetTasks
from qsprpred.plotting.interfaces import ModelPlot
from sklearn import metrics


class RegressionPlot(ModelPlot, ABC):

    def getSupportedTasks(self):
        return [TargetTasks.REGRESSION]


class CorrelationPlot(RegressionPlot):

    def make(self, save: bool = True, show: bool = False, out_dir: str = ".", filename_prefix: str = "corrplot"):
        """
        Function to plot the results of regression models. Plot predicted pX vs real pX.
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
                plt.scatter(df.Label, df.Score, s=5, label=legend, color=my_cmap[j])
                coef = metrics.r2_score(df.Label, df.Score)
                rmse = metrics.mean_squared_error(df.Label, df.Score, squared=False)
                summary["R2"].append(coef)
                summary["RMSE"].append(rmse)
                summary["Set"].append(cate_names[j])
                summary["ModelName"].append(model.name)

                plt.title(model)
                plt.xlabel(f"Experimental {model.targetProperty}")
                plt.ylabel(f"Predicted {model.targetProperty}")
                min_val_now = math.floor(min(pd.concat([df.Label, df.Score])))
                max_val_now = math.ceil(max(pd.concat([df.Label, df.Score])))
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
