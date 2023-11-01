"""
This is a simple script to check that the results of the tutorial match the
expected results. It is not part of the tutorial itself, but is used by the
CI/CD pipeline to check that the tutorial is up-to-date and that the models
are consistent with previous ones.
"""

import pandas as pd
import os
import sys


success = True
for f in os.listdir("expected"):
    for type in ["ind", "cv"]:
        try:
            file_name = f"{f}.{type}.tsv"
            print(f"Comparing file contents of {file_name}")
            relative_file_path = f"{f}/{file_name}"

            expected_file_path = f"expected/{relative_file_path}"
            actual_file_path = f"qspr/models/{relative_file_path}"

            expected_values = pd.read_csv(expected_file_path, sep="\t").set_index("QSPRID", drop=True).sort_index()
            actual_values = pd.read_csv(actual_file_path, sep="\t").set_index("QSPRID", drop=True).sort_index()
            expected_values = expected_values.round(2)
            actual_values = actual_values.round(2)
            assert expected_values.columns.equals(actual_values.columns), f"Column names do not match for file {file_name}."
            assert expected_values.index.equals(actual_values.index), f"Index values do not match for file {file_name}."
            assert expected_values.equals(actual_values), f"Values do not match for file {file_name}."
        except AssertionError as e:
            # show traceback and log error
            import traceback
            traceback.print_exc()
            sys.stderr.write(f"Comparison error in: {f}\n")
            success = False
            continue
        print("Comparison successful!")

if not success:
    sys.exit(1)
else:
    print("Comparison of tutorial outputs successful!")
