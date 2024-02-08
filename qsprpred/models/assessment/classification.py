import pandas as pd
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
)


def create_metrics_summary(model):
    decision_threshold: float = 0.5
    metrics = [
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        accuracy_score,
    ]
    summary = {"Metric": [], "Model": [], "TestSet": [], "Value": []}
    property_name = model.targetProperties[0].name

    cv_path = f"{model.outPrefix}.cv.tsv"
    ind_path = f"{model.outPrefix}.ind.tsv"

    df = pd.read_table(cv_path)

    # cross-validation
    for fold in sorted(df.Fold.unique()):
        y_pred = df[f"{property_name}_ProbabilityClass_1"][df.Fold == fold]
        y_pred_values = [1 if x > decision_threshold else 0 for x in y_pred]
        y_true = df[f"{property_name}_Label"][df.Fold == fold]
        for metric in metrics:
            val = metric(y_true, y_pred_values)
            summary["Metric"].append(metric.__name__)
            summary["Model"].append(model.name)
            summary["TestSet"].append(f"CV{fold + 1}")
            summary["Value"].append(val)

    # independent test set
    df = pd.read_table(ind_path)
    y_pred = df[f"{property_name}_ProbabilityClass_1"]
    th = 0.5
    y_pred_values = [1 if x > th else 0 for x in y_pred]
    y_true = df[f"{property_name}_Label"]
    for metric in metrics:
        val = metric(y_true, y_pred_values)
        summary["Metric"].append(metric.__name__)
        summary["Model"].append(model.name)
        summary["TestSet"].append("IND")
        summary["Value"].append(val)

    return summary
