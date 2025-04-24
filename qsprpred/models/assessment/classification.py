import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def create_metrics_summary(model, assessments):
    decision_threshold: float = 0.5
    metrics = [
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        accuracy_score,
    ]
    summary = {"Model": [], "Metric": [], "Assessment": [], "Fold": [], "Set": [], "Value": []}
    property_name = model.targetProperties[0].name

    assessment_paths = [f"{model.outPrefix}_{assessment}.tsv" for assessment in assessments]

    for metric in metrics:
        for assessment, assessment_path in zip(assessments, assessment_paths):
            df = pd.read_table(assessment_path)
            for fold in sorted(df.Fold.unique()):
                df_fold = df[df.Fold == fold]
                for set_name in ["Train", "Test"]:
                    df_set = df_fold[df_fold.Set == set_name]
                    y_pred = df_set[f"{property_name}_ProbabilityClass_1"]
                    y_pred_values = [1 if x > decision_threshold else 0 for x in y_pred]
                    y_true = df_set[f"{property_name}_Label"]
                    val = metric(y_true, y_pred_values)
                    summary["Model"].append(model.name)
                    summary["Metric"].append(metric.__name__)
                    summary["Assessment"].append(assessment)
                    summary["Set"].append(set_name)
                    summary["Fold"].append(fold)
                    summary["Value"].append(val)

    return summary
