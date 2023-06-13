"""
data

Created by: Martin Sicho
On: 12.05.23, 17:14
"""
from typing import Callable

import pandas as pd

from qsprpred.logs import logger

from ...data.data import MoleculeTable, QSPRDataset, TargetProperty
from ...data.utils.descriptorcalculator import (
    DescriptorsCalculator,
    MoleculeDescriptorsCalculator,
)
from ..data.utils.descriptorcalculator import ProteinDescriptorCalculator


class PCMDataset(QSPRDataset):
    """Construct QSPRdata, also apply transformations of output property if
    specified.

    arguments:
        name (str): data name, used in saving the data
        proteincol (str, optional): name of column in df containing the protein target
            identifier (usually a UniProt ID) to use for protein descriptors for PCM
            modelling and other protein related tasks. Defaults to None.
        proteinseqprovider: Callable = None, optional): function that takes a
            "proteincol" value and returns the appropriate protein sequence.
            Defaults to None.
        target_props (list[TargetProperty, dict]): target properties, names
            should correspond with target columnname in df
        df (pd.DataFrame, optional): input dataframe containing smiles and target
            property. Defaults to None.
        smilescol (str, optional): name of column in df containing SMILES.
            Defaults to "SMILES".
        add_rdkit (bool, optional): if true, column with rdkit molecules will be
            added to df. Defaults to False.
        store_dir (str, optional): directory for saving the output data.
            Defaults to '.'.
        overwrite (bool, optional): if already saved data at output dir if should be
            overwritten. Defaults to False.
        n_jobs (int, optional): number of parallel jobs. If <= 0, all available
            cores will be used. Defaults to 1.
        chunk_size (int, optional): chunk size for parallel processing.
            Defaults to 50.
        drop_invalids (bool, optional): if true, invalid SMILES will be dropped.
            Defaults to True.
        drop_empty (bool, optional): if true, rows with empty target property will
            be removed.
        target_imputer (Callable, optional): imputer for missing target property
            values. Defaults to None.
        index_cols (list[str], optional): columns to be used as index in the
            dataframe. Defaults to `None` in which case a custom ID will be
            generated.

    Raises:
        `ValueError`: Raised if threshold given with non-classification task.
    """
    def __init__(
        self,
        name: str,
        proteincol,
        target_props: list[TargetProperty | dict],
        df: pd.DataFrame = None,
        smilescol: str = "SMILES",
        proteinseqprovider: Callable = None,
        add_rdkit: bool = False,
        store_dir: str = ".",
        overwrite: bool = False,
        n_jobs: int = 1,
        chunk_size: int = 50,
        drop_invalids: bool = True,
        drop_empty: bool = True,
        target_imputer: Callable = None,
        index_cols: list[str] = None,
    ):
        """Construct QSPRdata, also apply transformations of output property if
        specified.

        Args:
            name (str): data name, used in saving the data
            proteincol (str, optional): name of column in df containing the protein
                target identifier (usually a UniProt ID) to use for protein descriptors
                for PCM modelling and other protein related tasks. Defaults to None.
            proteinseqprovider: Callable = None, optional): function that takes a
                "proteincol" value and returns the appropriate protein sequence.
                Defaults to None.
            target_props (list[TargetProperty | dict]): target properties, names
                should correspond with target columnname in df
            df (pd.DataFrame, optional): input dataframe containing smiles and target
                property. Defaults to None.
            smilescol (str, optional): name of column in df containing SMILES.
                Defaults to "SMILES".
            add_rdkit (bool, optional): if true, column with rdkit molecules will be
                added to df. Defaults to False.
            store_dir (str, optional): directory for saving the output data.
                Defaults to '.'.
            overwrite (bool, optional): if already saved data at output dir if should be
                overwritten. Defaults to False.
            n_jobs (int, optional): number of parallel jobs. If <= 0, all available
                cores will be used. Defaults to 1.
            chunk_size (int, optional): chunk size for parallel processing.
                Defaults to 50.
            drop_invalids (bool, optional): if true, invalid SMILES will be dropped.
                Defaults to True.
            drop_empty (bool, optional): if true, rows with empty target property will
                be removed.
            target_imputer (Callable, optional): imputer for missing target property
                values. Defaults to None.
            index_cols (list[str], optional): columns to be used as index in the
                dataframe. Defaults to `None` in which case a custom ID will be
                generated.

        Raises:
            `ValueError`: Raised if threshold given with non-classification task.
        """
        super().__init__(
            name,
            df=df,
            smilescol=smilescol,
            add_rdkit=add_rdkit,
            store_dir=store_dir,
            overwrite=overwrite,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            drop_invalids=drop_invalids,
            index_cols=index_cols,
            target_props=target_props,
            target_imputer=target_imputer,
            drop_empty=drop_empty,
        )
        self.proteincol = proteincol
        self.proteinseqprovider = proteinseqprovider

    def addProteinDescriptors(
        self,
        calculator: ProteinDescriptorCalculator,
        recalculate=False,
        featurize=True
    ):
        """
        Add protein descriptors to the data frame.

        Args:
            calculator (ProteinDescriptorCalculator): DescriptorsCalculator to use.
            recalculate (bool): Whether to recalculate descriptors even if they are
                already present in the data frame.
            featurize (bool): Whether to featurize the descriptors after adding them to
                the data frame.
        """

        if recalculate:
            self.dropDescriptors(calculator)
        elif self.getDescriptorNames(prefix=calculator.getPrefix()):
            logger.warning(
                f"Protein descriptors already exist in {self.name}. "
                "Use `recalculate=True` to overwrite them."
            )
            return

        if not self.proteincol:
            raise ValueError(
                "Protein column not set. Cannot calculate protein descriptors."
            )

        sequences, info = (
            self.proteinseqprovider(self.df[self.proteincol].unique().tolist())
            if self.proteinseqprovider else (None, {})
        )
        descriptors = calculator(self.df[self.proteincol].unique(), sequences, **info)
        descriptors[self.proteincol] = descriptors.index.values

        # add the descriptors to the descriptor list
        self.attachDescriptors(calculator, descriptors, [self.proteincol])
        self.featurize(update_splits=featurize)

    @staticmethod
    def fromSDF(name, filename, smiles_prop, *args, **kwargs):
        raise NotImplementedError(
            f"SDF loading not implemented for {PCMDataset.__name__}, yet."
            " You can convert from 'PCMTable' with 'fromMolTable'."
        )

    @staticmethod
    def fromMolTable(
        mol_table: MoleculeTable,
        proteincol,
        target_props: list[TargetProperty | dict],
        name=None,
        **kwargs,
    ):
        kwargs["store_dir"] = (
            mol_table.storeDir if "store_dir" not in kwargs else kwargs["store_dir"]
        )
        name = mol_table.name if name is None else name
        ds = PCMDataset(
            name, proteincol, target_props=target_props, df=mol_table.getDF(), **kwargs
        )
        if not ds.descriptors and mol_table.descriptors:
            ds.descriptors = mol_table.descriptors
            ds.descriptorCalculators = mol_table.descriptorCalculators
        return ds

    def addFeatures(
        self,
        feature_calculators: list[DescriptorsCalculator] = None,
        recalculate=False
    ):
        for calc in feature_calculators:
            if isinstance(calc, MoleculeDescriptorsCalculator):
                self.addDescriptors(calc, recalculate=recalculate, featurize=False)
            elif isinstance(calc, ProteinDescriptorCalculator):
                self.addProteinDescriptors(
                    calc, recalculate=recalculate, featurize=False
                )
            else:
                raise ValueError("Unknown feature calculator type: %s" % type(calc))
