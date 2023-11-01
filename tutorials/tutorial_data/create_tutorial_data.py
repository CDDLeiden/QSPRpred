import os

from qsprpred.data.data import QSPRDataset
from qsprpred.data.sources.papyrus import Papyrus
from qsprpred.models.tasks import TargetTasks


def A2AR(data_dir="tutorial_data", random_state=None):
    """A classification dataset that contains activity data on the adenosine A2A
    receptor loaded from the Papyrus database using the built-in Papyrus wrapper.

    Returns:
        a `QSPRDataset` instance with the loaded data
    """
    acc_keys = [
        "P29274"
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
        acc_keys, quality, name=dataset_name, use_existing=True, overwrite=True
    )

    print(f"Number of samples loaded: {len(dataset.getDF())}")

    df = dataset.getDF()
    df = df[["SMILES", "pchembl_value_Mean"]]
    df.to_csv(
        os.path.join(data_dir, "A2A_LIGANDS.tsv"), index=False, sep="\t"
    )

    return QSPRDataset.fromMolTable(
        dataset,
        [
            {
                "name": "pchembl_value_Median",
                "task": TargetTasks.SINGLECLASS,
                "th": [6.5],
            }
        ],
        overwrite=True,
        random_state=random_state,
    )


if __name__ == "__main__":
    A2AR()
