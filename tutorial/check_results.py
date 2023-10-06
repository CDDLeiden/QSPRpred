"""
This is a simple script to check that the results of the tutorial match the
expected results. It is not part of the tutorial itself, but is used by the
CI/CD pipeline to check that the tutorial is up-to-date and that the models
are consistent with previous ones.
"""

import hashlib
import os
import sys


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


success = True
for f in os.listdir("expected"):
    for type in ["ind", "cv"]:
        file_name = f"{f}.{type}.tsv"
        print(f"Comparing file contents of {file_name}")
        relative_file_path = f"{f}/{file_name}"

        expected_file_path = f"expected/{relative_file_path}"
        actual_file_path = f"qspr/models/{relative_file_path}"

        expected_hash = md5(expected_file_path)
        actual_hash = md5(actual_file_path)

        if expected_hash != actual_hash:
            with open(actual_file_path) as file:
                success = False
                print(
                    f"Failure: contents of {relative_file_path} do not match.\
                      Actual content:"
                )
                print(file.read())

if not success:
    sys.exit(1)
else:
    print("Comparison of tutorial output successful!")
