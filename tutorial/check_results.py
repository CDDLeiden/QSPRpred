import hashlib
import os

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

succes = True
for f in os.listdir("expected"):
    for type in ['ind', 'cv']:
        relative_file_path = f + "/" + f + "." + type + ".tsv"

        expected_file_path = "expected/" + relative_file_path
        actual_file_path = "qspr/models/" + relative_file_path

        expected_hash = md5(expected_file_path)
        actual_hash = md5(actual_file_path)

        if expected_hash != actual_hash:
            with open(actual_file_path) as file:
                success = False
                print("Failure: contents of " + relative_file_path + " do not match. Actual content:")
                print(file.read())
            
if not succes:
    exit(1)