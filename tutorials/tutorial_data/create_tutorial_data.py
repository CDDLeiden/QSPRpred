import os

from qsprpred.data.sources.papyrus import Papyrus
import argparse


def SingleTaskTutorialData(data_dir: str | None = None):
    """Creates a dataset that contains activity data on the adenosine A2A
    receptor loaded from the Papyrus database using the built-in Papyrus wrapper.

    Args:
        data_dir: directory where the dataset will be saved, if None, the
            tutorial_data directory will be used

    Returns:
        a `MolTable` instance with the loaded data
    """
    data_dir = os.path.dirname(__file__) if data_dir is None else data_dir
    acc_keys = [
        "P29274",  # A2A
    ]  # Adenosine receptor A2A (https://www.uniprot.org/uniprotkb/P29274/entry)
    dataset_name = "A2A_LIGANDS"  # name of the file to be generated
    quality = "high"  # choose minimum quality from {"high", "medium", "low"}
    papyrus_version = "05.6"  # Papyrus database version

    papyrus = Papyrus(
        data_dir=data_dir,
        stereo=False,
        version=papyrus_version,
        plus_only=True,
    )

    dataset = papyrus.getData(
        dataset_name, acc_keys, quality, use_existing=True, overwrite=True
    )

    print(f"Number of samples loaded: {len(dataset.getDF())}")

    # Save the dataset as a .tsv file
    df = dataset.getDF()
    df = df[["SMILES", "pchembl_value_Mean", "Year"]]

    df.to_csv(os.path.join(data_dir, "A2A_LIGANDS.tsv"), index=False, sep="\t")

    return dataset


def MultiTaskTutorialData(data_dir: str | None = None):
    """Creates a dataset that contains activity data on the adenosine
    receptors (A1, A2A, A2B and A3) loaded from the Papyrus database using
    the built-in Papyrus wrapper.

    Args:
        data_dir: directory where the dataset will be saved, if None, the
            tutorial_data directory will be used

    Returns:
        a `MolTable` instance with the loaded data
    """
    data_dir = os.path.dirname(__file__) if data_dir is None else data_dir
    acc_keys = [
        "P30542",  # A1
        "P29274",  # A2A
        "P29275",  # A2B
        "P0DMS8",  # A3
    ]  # Adenosine receptor A2A (https://www.uniprot.org/uniprotkb/P29274/entry)
    dataset_name = "AR_LIGANDS"  # name of the file to be generated
    quality = "high"  # choose minimum quality from {"high", "medium", "low"}
    papyrus_version = "05.6"  # Papyrus database version

    papyrus = Papyrus(
        data_dir=data_dir,
        stereo=False,
        version=papyrus_version,
        plus_only=True,
    )

    dataset = papyrus.getData(
        dataset_name, acc_keys, quality, use_existing=True, overwrite=True
    )

    print(f"Number of samples loaded: {len(dataset.getDF())}")

    # Save the dataset as a .tsv file
    df = dataset.getDF()
    df = df[["SMILES", "pchembl_value_Mean", "accession"]]

    df.to_csv(os.path.join(data_dir, "AR_LIGANDS.tsv"), index=False, sep="\t")

    return dataset

def prepare_multitask(file_path: str):
    import pandas as pd
    df = pd.read_csv(file_path, sep="\t")
    df = df.pivot(index="SMILES", columns="accession", values="pchembl_value_Mean")
    df.columns.name = None
    df.reset_index(inplace=True)
    df.to_csv(f"{file_path.split('.')[0]}_pivot.tsv", sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create tutorial data")
    parser.add_argument(
        "-p",
        "--pivot",
        type=str,
        help="Pivot AR_ligands dataset, provide the file path")

    args = parser.parse_args()

    if args.pivot is not None:
        prepare_multitask(args.pivot)
    else:
        SingleTaskTutorialData()
        MultiTaskTutorialData()
