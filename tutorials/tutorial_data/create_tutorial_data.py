import os

from qsprpred.data.sources.papyrus import Papyrus


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


if __name__ == "__main__":
    SingleTaskTutorialData()
    MultiTaskTutorialData()
