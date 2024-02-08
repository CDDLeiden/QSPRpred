import pandas as pd

from sklearn import metrics


def create_correlation_summary(model):
    cv_path = f"{model.outPrefix}.cv.tsv"
    ind_path = f"{model.outPrefix}.ind.tsv"

    cate = [cv_path, ind_path]
    cate_names = ["cv", "ind"]
    property_name = model.targetProperties[0].name
    summary = {"ModelName": [], "R2": [], "RMSE": [], "Set": []}
    for j, _ in enumerate(["Cross Validation", "Independent Test"]):
        df = pd.read_table(cate[j])
        coef = metrics.r2_score(
            df[f"{property_name}_Label"], df[f"{property_name}_Prediction"]
        )
        rmse = metrics.root_mean_squared_error(
            df[f"{property_name}_Label"],
            df[f"{property_name}_Prediction"],
        )
        summary["R2"].append(coef)
        summary["RMSE"].append(rmse)
        summary["Set"].append(cate_names[j])
        summary["ModelName"].append(model.name)

    return summary
