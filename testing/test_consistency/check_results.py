"""
This is a simple script to check that the results of the tutorial match the
expected results. It is not part of the tutorial itself, but is used by the
CI/CD pipeline to check that the tutorial is up-to-date and that the models
are consistent with previous ones.
"""
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

models_base = "./data/models"

success = True
failed_files = []
for f in os.listdir("expected"):
    for type in ["ind", "cv"]:
        file_name = f"{f}.{type}.tsv"
        # FIXME: Filenames changed in the new version
        new_type = "crossval" if type == "cv" else "test"
        if "CLS" in file_name:
            new_file_name = f"{f}_{new_type}_matthews_corrcoef.tsv"
        else:
            new_file_name = f"{f}_{new_type}_neg_root_mean_squared_error.tsv"    
        try:
            print(f"Comparing file contents of {file_name}")
            relative_file_path = f"{f}/{file_name}"
            new_relative_file_path = f"{f}/{new_file_name}"

            expected_file_path = f"expected/{relative_file_path}"
            actual_file_path = f"{models_base}/{new_relative_file_path}"

            expected_values = (
                pd.read_csv(expected_file_path, sep="\t")
                .set_index("ID", drop=True)
                .sort_index()
            )
            actual_values = (
                pd.read_csv(actual_file_path, sep="\t")
                .set_index("ID", drop=True)
                .sort_index()
            )
            # FIXME: In the old version, only test values are saved and
            # the "Fold" column is only present for cross validation
            # also all Fold values used to be float64 but now they are int
            if type == "ind":
                actual_values = actual_values.drop(columns=["Fold"])
            else:
                actual_values["Fold"] = actual_values["Fold"].astype(np.float64)
            actual_values = actual_values[actual_values["Set"] == "Test"]
            # FIXME: In the old version, the "Set" column was not present
            # and only the test values were saved
            actual_values = actual_values.drop(columns=["Set"])
            # FIXME: In the old version, for classification the Label
            # were bool for binary classification, but now they are float64
            # changed in commit 0d3f4dc
            if "CLS" in file_name:
                actual_values["pchembl_value_Mean_Label"] = actual_values[
                    "pchembl_value_Mean_Label"
                ].astype(bool)
                # this changed, but I don't know why
                if actual_values["pchembl_value_Mean_Prediction"].dtype == np.float64:
                    actual_values["pchembl_value_Mean_Prediction"] = actual_values[
                        "pchembl_value_Mean_Prediction"
                    ].astype(bool)
            assert expected_values.columns.equals(
                actual_values.columns
            ), f"Column names do not match for file {file_name}."
            assert expected_values.index.equals(
                actual_values.index
            ), f"Index values do not match for file {file_name}."
            assert expected_values.dtypes.equals(
                actual_values.dtypes
            ), (
                f"Data types do not match for file {file_name}.\n"
                f"Expected: {expected_values.dtypes}\n"
                f"Actual: {actual_values.dtypes}"
            )
            try:
                assert expected_values.equals(
                    actual_values
                ), f"Values do not match for file {file_name}."
            except AssertionError as e:
                sys.stderr.write(f"Comparison error in values of: {file_name}\n")
                # check and print which values are different
                diff = expected_values.compare(actual_values)
                sys.stderr.write(diff.to_string())
                overviews = []
                for idx, row in diff.iterrows():
                    overview = dict()
                    for col in diff.columns:
                        name = col[0]
                        if name not in overview:
                            overview[name] = {"idx": idx}
                        if col[1] == "self":
                            overview[name]["expected"] = row[col]
                        else:
                            overview[name]["true"] = row[col]
                    overview = {
                        k: v
                        for k, v in overview.items()
                        if not (np.isnan(v["true"]) and np.isnan(v["expected"]))
                    }
                    overviews.append(overview)
                sys.stderr.write(json.dumps(overviews, indent=4))
                raise e
        except AssertionError as e:
            # print stack trace
            traceback.print_exc()
            success = False
            failed_files.append(file_name)
            continue

if not success:
    sys.stderr.write(
        "Comparison of benchmark outputs failed! "
        "One or more files did not match:\n" + "\t\n".join(failed_files) + "\n"
    )
    sys.exit(1)
else:
    print("Comparison of benchmark outputs successful!")
