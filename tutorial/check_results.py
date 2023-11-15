"""
This is a simple script to check that the results of the tutorial match the
expected results. It is not part of the tutorial itself, but is used by the
CI/CD pipeline to check that the tutorial is up-to-date and that the models
are consistent with previous ones.
"""
import json

import numpy as np
import pandas as pd
import os
import sys
import traceback


success = True
for f in os.listdir("expected"):
    for type in ["ind", "cv"]:
        try:
            file_name = f"{f}.{type}.tsv"
            print(f"Comparing file contents of {file_name}")
            relative_file_path = f"{f}/{file_name}"

            expected_file_path = f"expected/{relative_file_path}"
            actual_file_path = f"qspr/models/{relative_file_path}"

            expected_values = (
                pd.read_csv(expected_file_path, sep="\t")
                .set_index("QSPRID", drop=True)
                .sort_index()
            )
            actual_values = (
                pd.read_csv(actual_file_path, sep="\t")
                .set_index("QSPRID", drop=True)
                .sort_index()
            )
            assert expected_values.columns.equals(
                actual_values.columns
            ), f"Column names do not match for file {file_name}."
            assert expected_values.index.equals(
                actual_values.index
            ), f"Index values do not match for file {file_name}."
            try:
                assert expected_values.equals(
                    actual_values
                ), f"Values do not match for file {file_name}."
            except AssertionError as e:
                sys.stderr.write(f"Comparison error in values of: {file_name}\n")
                # check and print which values are different
                diff = expected_values.compare(actual_values)
                overviews = []
                for idx, row in diff.iterrows():
                    overview = dict()
                    for col in diff.columns:
                        name = col[0]
                        if name not in overview:
                            overview[name] = {"idx": idx}
                        if col[1] == "self":
                            overview[name]["true"] = row[col]
                        else:
                            overview[name]["expected"] = row[col]
                    overview = {
                        k: v for k, v in overview.items()
                        if not (np.isnan(v["true"]) and np.isnan(v["expected"]))
                    }
                    overviews.append(overview)
                sys.stderr.write(json.dumps(overviews, indent=4))
                raise e
        except AssertionError as e:
            # print  stack trace
            traceback.print_exc()
            success = False
            continue
        print("Comparison successful!")

if not success:
    print("Comparison of tutorial outputs failed! One or more files did not match.")
    sys.exit(1)
else:
    print("Comparison of tutorial outputs successful!")