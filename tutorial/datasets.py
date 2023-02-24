"""
Module with some example datasets.

Created by: Martin Sicho
On: 11.01.23, 15:17
"""
import os

import numpy as np
import pandas as pd
from qsprpred.data.data import QSPRDataset
from qsprpred.data.sources.papyrus import Papyrus
from qsprpred.models.tasks import TargetTasks


def A2AR(data_dir='data'):
    """A classification dataset that contains activity data on the adenosine A2A receptor loaded from the Papyrus database
    using the built-in Papyrus wrapper.

    Returns:
        a `QSPRDataset` instance with the loaded data
    """
    acc_keys = ["P29274"]  # Adenosine receptor A2A (https://www.uniprot.org/uniprotkb/P29274/entry)
    dataset_name = "A2A_LIGANDS"  # name of the file to be generated
    quality = "high"  # choose minimum quality from {"high", "medium", "low"}
    papyrus_version = '05.6'  # Papyrus database version

    papyrus = Papyrus(
        data_dir=data_dir,
        stereo=False,
        version=papyrus_version
    )

    dataset = papyrus.getData(
        acc_keys,
        quality,
        name=dataset_name,
        use_existing=True,
        store_dir="qspr/data"
    )

    print(f"Number of samples loaded: {len(dataset.getDF())}")
    return QSPRDataset.fromMolTable(
        dataset, [{"name": "pchembl_value_Median", "task": TargetTasks.SINGLECLASS, "th": [6.5]}])


def Parkinsons():
    """Parkinson's disease dataset that contains data for multiple targets related to the disease.

    It is loaded from a CSV file into pandas `DataFrame`, which is then converted to `QSPRDataset`
    regression data set with 'GABAAalpha' activity as the target property.

    Returns:
        a `QSPRDataset` instance with the loaded data
    """
    os.makedirs('qspr/data', exist_ok=True)

    # Load in the data
    df = pd.read_csv('data/parkinsons_dp_original.csv', sep=',')

    smiles_col = 'SMILES'
    activity_col = 'pchembl_value_Mean'
    target_col = 'accession'

    # combine uniprot accessions of same protein
    df = df.loc[df['accession'].isin(
        ['P14867', 'P31644', 'P34903', 'P47869', 'P48169', 'Q16445', 'O15399', 'O60391', 'Q05586', 'Q12879', 'Q13224',
         'Q14957', 'Q8TCU5', 'Q14643', 'P41594', 'Q13255'])]
    df.loc[
        df['accession'].isin(['P14867', 'P31644', 'P34903', 'P47869', 'P48169', 'Q16445']), 'accession'] = 'GABAAalpha'
    df.loc[df['accession'].isin(
        ['O15399', 'O60391', 'Q05586', 'Q12879', 'Q13224', 'Q14957', 'Q8TCU5']), 'accession'] = 'NMDA'

    # drop columns without pchembl value
    df = df.dropna(subset=['pchembl_value_Mean'])

    # print number of samples per target
    print("Number of samples per target:")
    print(df[target_col].value_counts())

    # Get data in correct format and taking the mean if multiple activatie values per smiles
    df = df.pivot_table(index=[smiles_col], columns=[target_col], values=activity_col, aggfunc=np.mean).reset_index()
    # df.to_csv('data/parkinsons_pivot.tsv', sep='\t', index=False)

    return QSPRDataset(
        name='tutorial_data',
        df=df,
        smilescol=smiles_col,
        target_props=[{"name": "GABAAalpha", "task": TargetTasks.REGRESSION}],
        store_dir="qspr/data"
    )
