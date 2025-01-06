import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error


def create_correlation_summary(model, assessments):
    assessment_paths = [f"{model.outPrefix}_{assessment}.tsv" for assessment in assessments]

    property_name = model.targetProperties[0].name
    summary = {"Model": [], "Metric": [], "Assessment": [], "Fold": [], "Set": [], "Value": []}
    for assessment, assessment_path in zip(assessments, assessment_paths):
        df = pd.read_table(assessment_path)
        for metric in [r2_score, root_mean_squared_error]:
            for fold in sorted(df.Fold.unique()):
                df_fold = df[df.Fold == fold]
                for set_name in ["Train", "Test"]:
                    df_set = df_fold[df_fold.Set == set_name]
                    y_true = df_set[f"{property_name}_Label"]
                    y_pred = df_set[f"{property_name}_Prediction"]
                    val = metric(y_true, y_pred)
                    summary["Model"].append(model.name)
                    summary["Metric"].append(metric.__name__)
                    summary["Assessment"].append(assessment)
                    summary["Fold"].append(fold)
                    summary["Set"].append(set_name)
                    summary["Value"].append(val)

    return summary
