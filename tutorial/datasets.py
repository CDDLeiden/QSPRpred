"""
Module with some example datasets.

Created by: Martin Sicho
On: 11.01.23, 15:17
"""
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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
        use_existing=True
    )

    print(f"Number of samples loaded: {len(dataset.getDF())}")
    return QSPRDataset.fromMolTable(
        dataset, [{"name": "pchembl_value_Median", "task": TargetTasks.SINGLECLASS, "th": [6.5]}])


def Parkinsons(singletask=True):
    """Parkinson's disease dataset that contains data for multiple targets related to the disease.

    It is loaded from a CSV file into pandas `DataFrame`. This is then converted to `QSPRDataset`
    regression data set with 'GABAAalpha' activity as the target property for single task & the 
    mGLU receptors for multitask.

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
         'Q14957', 'Q8TCU5', 'Q14643', 'O00222', 'O15303', 'P41594', 'Q13255', 'Q14416', 'Q14831', 'Q14832', 'Q14833'])]
    df.loc[
        df['accession'].isin(['P14867', 'P31644', 'P34903', 'P47869', 'P48169', 'Q16445']), 'accession'] = 'GABAAalpha'
    df.loc[df['accession'].isin(
        ['O15399', 'O60391', 'Q05586', 'Q12879', 'Q13224', 'Q14957', 'Q8TCU5']), 'accession'] = 'NMDA'

    # drop columns without pchembl value
    df = df.dropna(subset=['pchembl_value_Mean'])

    if singletask:
        # print number of samples per target
        print("Number of samples per target:")
        print(df[target_col].value_counts())

    # Get data in correct format and taking the mean if multiple activatie values per smiles
    df = df.pivot_table(index=[smiles_col], columns=[target_col], values=activity_col, aggfunc=np.mean).reset_index()
    df.to_csv('data/parkinsons_pivot.tsv', sep='\t', index=False)

    if singletask:
        return QSPRDataset(
            name='tutorial_data',
            df=df,
            smilescol=smiles_col,
            target_props=[{"name": "GABAAalpha", "task": TargetTasks.REGRESSION}],
            store_dir="qspr/data"
        )

    else:
        target_props = []
        # for target in list of mGLU receptors
        for target in ['O00222', 'O15303', 'P41594', 'Q13255', 'Q14416', 'Q14831', 'Q14832', 'Q14833']:
            target_props.append({"name": target, "task": TargetTasks.REGRESSION})
            
        return QSPRDataset(
            name='tutorial_data',
            df=df,
            smilescol=smiles_col,
            target_props=target_props,
            store_dir="qspr/data",
            target_imputer=SimpleImputer(strategy='mean'),
            overwrite=True
        )

def AR_PCM(data_dir='data'):
    """
    A classification dataset that contains activity data for a PCM approach to model activity for a selection of adenosine receptors. The function recreates steps from data_preparation_advanced.ipynb.

    Returns:
        a `QSPRDataset` instance with the loaded data
    """

    acc_keys = ["P29274", "P29275", "P30542", "P0DMS8"]
    dataset_name = "AR_LIGANDS"  # name of the file to be generated
    quality = "high"  # choose minimum quality from {"high", "medium", "low"}
    papyrus_version = '05.6'  # Papyrus database version

    papyrus = Papyrus(
        data_dir=data_dir,
        stereo=False,
        version=papyrus_version
    )

    mt = papyrus.getData(
        acc_keys,
        quality,
        name=dataset_name,
        use_existing=True
    )
    ds_seq = papyrus.getProteinData(acc_keys, name=f"{mt.name}_seqs", use_existing=True)

    def sequence_provider(acc_keys):
        """
        A function that provides a mapping from accession key to a protein sequence.

        Args:
            acc_keys (list): Accession keys of the protein to get a mapping of sequences for.

        Returns:
            (dict) : Mapping of accession keys to protein sequences.
            (dict) : Additional information to pass to the MSA provider (can be empty).
        """
        map = dict()
        info = dict()
        for i, row in ds_seq.iterrows():
            map[row['accession']] = row['Sequence']

            # can be omitted
            info[row['accession']] = {
                'Organism': row['Organism'],
                'UniProtID': row['UniProtID'],
            }

        return map, info

    return QSPRDataset.fromMolTable(
        mt,
        target_props=[
            {
                "name": "pchembl_value_Median",
                "task": TargetTasks.SINGLECLASS,
                "th": [6.5]
            }
        ],
        proteincol="accession",
        proteinseqprovider=sequence_provider,
    )