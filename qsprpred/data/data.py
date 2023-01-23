"""This module contains the QSPRDataset that holds and prepares data for modelling."""
import concurrent
import json
import os
import warnings
from typing import Callable, List, Literal

import numpy as np
import pandas as pd
from qsprpred.data.interfaces import MoleculeDataSet, datasplit
from qsprpred.data.utils.datasplitters import randomsplit
from qsprpred.data.utils.descriptorcalculator import Calculator, DescriptorsCalculator
from qsprpred.data.utils.feature_standardization import SKLearnStandardizer
from qsprpred.data.utils.scaffolds import Scaffold
from qsprpred.data.utils.smiles_standardization import (
    chembl_smi_standardizer,
    sanitize_smiles,
)
from qsprpred.logs import logger
from qsprpred.models.tasks import ModelTasks
from rdkit import Chem
from rdkit.Chem import PandasTools
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm


class MoleculeTable(MoleculeDataSet):
    """
    Class that holds and prepares molecule data for modelling and other analyses.
    """

    class ParallelApplyWrapper:
        """
        A wrapper class for parallelizing pandas apply functions.

        """
        def __init__(self, func, func_args=None, func_kwargs=None, axis=0, raw=False, result_type='expand'):
            """

            Initialize the instance with pandas parameters to apply to chunks of data.

            See `pandas.DataFrame.apply` for more information.

            Args:
                func: function to apply
                func_args: arguments to pass to func
                func_kwargs: keyword arguments to pass to func
                axis: axis to apply func on (0 for columns, 1 for rows)
                raw: whether to pass Series object to func or raw array
                result_type: whether to expand/ignore the results. See pandas.DataFrame.apply for more info.

            """
            self.args = func_args
            self.kwargs = func_kwargs
            self.func = func
            self.axis = axis
            self.raw = raw
            self.result_type = result_type

        def __call__(self, data: pd.DataFrame):
            """
            Apply the function to the current chunk of data.

            Args:
                data: chunk of data to apply function to

            Returns:
                result of applying function to chunk of data
            """

            return data.apply(self.func, raw=self.raw, axis=self.axis, result_type=self.result_type,
                              args=self.args, **self.kwargs if self.kwargs else {})

    def __init__(
            self,
            name : str,
            df : pd.DataFrame = None,
            smilescol : str = "SMILES",
            add_rdkit : bool = False,
            store_dir : str = '.',
            overwrite : bool = False,
            n_jobs : int = 0,
            chunk_size : int = 50,
    ):
        """

        Initialize a `MoleculeTable` object. This object wraps a pandas dataframe and provides short-hand methods to prepare molecule
        data for modelling and analysis.

        Args:
            name (str): Name of the dataset. You can use this name to load the dataset from disk anytime and create a new instance.
            df (pd.DataFrame): Pandas dataframe containing the data. If you provide a dataframe for a dataset that already exists on disk,
            the dataframe from disk will override the supplied data frame. Set 'overwrite' to `True` to override the data frame on disk.
            smilescol (str): Name of the column containing the SMILES sequences of molecules.
            add_rdkit (bool): Add RDKit molecule instances to the dataframe. WARNING: This can take a lot of memory.
            store_dir (str): Directory to store the dataset files. Defaults to the current directory. If it already contains files with the same name, the existing data will be loaded.
            overwrite (bool): Overwrite existing dataset.
            n_jobs (int): Number of jobs to use for parallel processing.
            chunk_size (int): Size of chunks to use per job in parallel processing.
        """

        # settings
        self.smilescol = smilescol
        self.includesRdkit = add_rdkit
        self.name = name
        self.descriptorCalculator = None
        self.nJobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.chunkSize = chunk_size

        # paths
        self.storeDir = store_dir.rstrip("/")
        self.storePrefix = f"{self.storeDir}/{self.name}"
        self.descriptorCalculatorPath = f"{self.storePrefix}_feature_calculators.json"
        if not os.path.exists(self.storeDir):
            raise FileNotFoundError(f"Directory '{self.storeDir}' does not exist.")
        self.storePath = f'{self.storePrefix}_df.pkl'

        # data frame initialization
        if df is not None:
            if self._isInStore('df') and not overwrite:
                warnings.warn(
                    'Existing data set found, but also found a data frame in store. Refusing to overwrite data. If you want to overwrite data in store, set overwrite=True.')
                self.reload()
            else:
                self.clear()
                self.df = df
                if self.includesRdkit:
                    PandasTools.AddMoleculeColumnToFrame(
                        self.df, smilesCol=self.smilescol, molCol='RDMol', includeFingerprints=False)
        else:
            if not self._isInStore('df'):
                raise ValueError(
                    f"No data frame found in store for '{self.name}'. Are you sure this is the correct dataset? If you are creating a new data set, make sure to supply a data frame.")
            self.reload()

    def __len__(self):
        """
        Returns:
            int: Number of molecules in the data set.
        """
        return len(self.df)

    def getDF(self):
        """
        Returns:
            pd.DataFrame: The data frame this instance manages.
        """

        return self.df

    def _isInStore(self, name):
        """
        Check if a pickled file with the given suffix exists.

        Args:
            name (str): Suffix of the file to check.

        Returns:
            bool: `True` if the file exists, `False` otherwise.
        """
        return os.path.exists(self.storePath) and self.storePath.endswith(f'_{name}.pkl')

    def save(self):
        """
        Save the data frame to disk and all associated files.

        Returns:
            str: Path to the saved data frame.
        """

        # save data frame
        self.df.to_pickle(self.storePath)

        # save descriptor calculator
        if self.descriptorCalculator:
            self.descriptorCalculator.toFile(self.descriptorCalculatorPath)

        return self.storePath

    def clear(self):
        """
        Remove all files associated with this data set from disk.
        """

        for file in os.listdir(self.storeDir):
            if file.startswith(self.name):
                os.remove(f'{self.storeDir}/{file}')

    def reload(self):
        """
        Reload the data table from disk.

        """

        self.df = pd.read_pickle(self.storePath)
        if os.path.exists(self.descriptorCalculatorPath):
            self.descriptorCalculator = DescriptorsCalculator.fromFile(self.descriptorCalculatorPath)

    @staticmethod
    def fromFile(filename, *args, **kwargs) -> 'MoleculeTable':
        """
        Create a `MoleculeTable` instance from by providing a direct path to the pickled data frame in storage.
        """

        store_dir = os.path.dirname(filename)
        name = os.path.basename(filename).split('.')[0]
        return MoleculeTable(name=name, store_dir=store_dir, *args, **kwargs)

    @staticmethod
    def fromSMILES(name, smiles, *args, **kwargs):
        """
        Create a `MoleculeTable` instance from a list of SMILES sequences.

        Args:
            name (str): Name of the data set.
            smiles (list): List of SMILES sequences.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: Additional keyword arguments to pass to the `MoleculeTable` constructor.
        """

        smilescol = "SMILES"
        df = pd.DataFrame({smilescol : smiles})
        return MoleculeTable(name, df, *args, smilescol=smilescol, **kwargs)

    @staticmethod
    def fromTableFile(name, filename, sep="\t", *args, **kwargs):
        """
        Create a `MoleculeTable` instance from a file containing a table of molecules (i.e. a CSV file).

        Args:
            name (str): Name of the data set.
            filename (str): Path to the file containing the table.
            sep (str): Separator used in the file for different columns.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: Additional keyword arguments to pass to the `MoleculeTable` constructor.
        """
        return MoleculeTable(name, pd.read_table(filename, sep=sep), *args, **kwargs)

    @staticmethod
    def fromSDF(name, filename, smiles_prop, *args, **kwargs):
        """
        Create a `MoleculeTable` instance from an SDF file.

        Args:
            name (str): Name of the data set.
            filename (str): Path to the SDF file.
            smiles_prop (str): Name of the property in the SDF file containing the SMILES sequence.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: Additional keyword arguments to pass to the `MoleculeTable` constructor.
        """
        return MoleculeTable(name, PandasTools.LoadSDF(filename, molColName="RDMol"), smilescol=smiles_prop, *args, **kwargs) # FIXME: in this case the RDKit molecule is always added, which can in most cases is an unnecessary overhead

    def getSubset(self, prefix: str):
        """
        Get a subset of the data set by providing a prefix for the column names or a column name directly.

        Args:
            prefix (str): Prefix of the column names to select.
        """

        if self.df.columns.str.startswith(prefix).any():
            return self.df[self.df.columns[self.df.columns.str.startswith(prefix)]]

    def apply(self, func, func_args=None, func_kwargs=None, axis=0, raw=False,
              result_type='expand', subset=None):
        """
        Apply a function to the data frame. In addition to the arguments of `pandas.DataFrame.apply`, this method also
        supports parallelization using `multiprocessing.Pool`.

        Args:
            func (callable): Function to apply to the data frame.
            func_args (list): Positional arguments to pass to the function.
            func_kwargs (dict): Keyword arguments to pass to the function.
            axis (int): Axis along which the function is applied (0 for column, 1 for rows).
            raw (bool): Whether to pass the data frame as-is to the function or to pass each row/column as a Series to the function.
            result_type (str): Whether to expand the result of the function to columns or to leave it as a Series.
            subset (list): List of column names if only a subset of the data should be used (reduces memory consumption).
        """
        n_cpus = self.nJobs
        chunk_size = self.chunkSize
        if n_cpus and n_cpus > 1:
            return self.papply(func, func_args, func_kwargs, axis, raw, result_type, subset, n_cpus, chunk_size)
        else:
            df_sub = self.df[subset if subset else self.df.columns]
            return df_sub.apply(func, raw=raw, axis=axis, result_type=result_type,
                                args=func_args, **func_kwargs if func_kwargs else {})

    def papply(self, func, func_args=None, func_kwargs=None, axis=0, raw=False,
               result_type='expand', subset=None, n_cpus=1, chunk_size=1000):
        """
        Parallelized version of `MoleculeTable.apply`.

        Args:
            func (callable): Function to apply to the data frame.
            func_args (list): Positional arguments to pass to the function.
            func_kwargs (dict): Keyword arguments to pass to the function.
            axis (int): Axis along which the function is applied (0 for column, 1 for rows).
            raw (bool): Whether to pass the data frame as-is to the function or to pass each row/column as a Series to the function.
            result_type (str): Whether to expand the result of the function to columns or to leave it as a Series.
            subset (list): List of column names if only a subset of the data should be used (reduces memory consumption).
            n_cpus (int): Number of CPUs to use for parallelization.
            chunk_size (int): Number of rows to process in each chunk.
            n_cpus (int): Number of CPUs to use for parallelization.
            chunk_size (int): Number of rows to process in each chunk.
        """

        n_cpus = n_cpus if n_cpus else os.cpu_count()
        df_sub = self.df[subset if subset else self.df.columns]
        data = [df_sub[i: i + chunk_size] for i in range(0, len(df_sub), chunk_size)]
        batch_size = n_cpus  # how many batches to prefetch into the process (more is faster, but uses more memory)
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as executor:
            batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
            for batch in tqdm(batches,
                              desc=f"Parallel apply in progress for {self.name}."):
                wrapped = self.ParallelApplyWrapper(
                    func,
                    func_args=func_args,
                    func_kwargs=func_kwargs,
                    result_type=result_type,
                    axis=axis,
                    raw=raw,
                )
                for result in executor.map(wrapped, batch):
                    results.append(result)

        return pd.concat(results, axis=0)

    def transform(self, targets, transformers, addAs=None):
        """
        Transform the data frame (or its part) using a list of transformers. Each transformer is a function that takes the data frame
        (or a subset of it as defined by the `targets` argument) and returns a transformed data frame. The transformed
        data frame can then be added to the original data frame if `addAs` is set to a `list` of new column names. If
        `addAs` is not `None`, the result of the application of transformers must have the same number of rows as the
        original data frame.

        Args:
            targets (list): List of column names to transform.
            transformers (list): List of transformers to apply to the columns.
        """

        ret = self.df[targets]
        for transformer in transformers:
            ret = transformer(ret)

        if not addAs:
            return ret
        else:
            self.df[addAs] = ret

    def filter(self, table_filters: List[Callable]):
        """
        Filter the data frame using a list of filters. Each filter is a function that takes the data frame and returns a
        a new data frame with the filtered rows. The new data frame is then used as the input for the next filter. The
        final data frame is saved as the new data frame of the `MoleculeTable`.
        """

        df_filtered = None
        for table_filter in table_filters:
            df_filtered = table_filter(self.df)

        if df_filtered is not None:
            self.df = df_filtered.copy()

    def addDescriptors(self, calculator: Calculator, recalculate=False):
        """
        Add descriptors to the data frame using a `Calculator` object.

        Args:
            calculator (Calculator): Calculator object to use for descriptor calculation.
            recalculate (bool): Whether to recalculate descriptors even if they are already present in the data frame.
                If `False`, existing descriptors are kept and no calculation takes place.
        """

        if recalculate:
            self.df.drop(self.getDescriptorNames(), axis=1, inplace=True)
        elif self.hasDescriptors:
            logger.warning(f"Descriptors already exist in {self.name}. Use `recalculate=True` to overwrite them.")
            return

        descriptors = self.apply(
            calculator,
            axis=0,
            subset=[
                self.smilescol],
            result_type='reduce'
        )
        descriptors = descriptors.to_list()
        descriptors = pd.concat(descriptors, axis=0)
        descriptors.index = self.df.index
        self.df = self.df.join(descriptors, how='left')
        self.descriptorCalculator = calculator

    def getDescriptors(self):
        """
        Get the subset of the data frame that contains only descriptors.

        Returns:
            pd.DataFrame: Data frame containing only descriptors.
        """

        return self.df[self.getDescriptorNames()]

    def getDescriptorNames(self):
        """
        Get the names of the descriptors in the data frame.

        Returns:
            list: List of descriptor names.
        """
        return [col for col in self.df.columns if col.startswith("Descriptor_")]

    @property
    def hasDescriptors(self):
        """
        Check whether the data frame contains descriptors.
        """
        return len(self.getDescriptorNames()) > 0

    def getProperties(self):
        """
        Get names of all properties/variables saved in the data frame (all columns).

        Returns:
            list: List of property names.
        """

        return self.df.columns

    def hasProperty(self, name):
        """
        Check whether a property is present in the data frame.

        Args:
            name (str): Name of the property.

        Returns:
            bool: Whether the property is present.
        """
        return name in self.df.columns

    def addProperty(self, name, data):
        """
        Add a property to the data frame.

        Args:
            name (str): Name of the property.
            data (list): List of property values.
        """
        self.df[name] = data

    def removeProperty(self, name):
        """
        Remove a property from the data frame.

        Args:
            name (str): Name of the property to delete.
        """
        del self.df[name]

    @staticmethod
    def _scaffold_calculator(mol, scaffold : Scaffold):
        return scaffold(mol[0])

    def addScaffolds(self, scaffolds: List[Scaffold], add_rdkit_scaffold=False):
        """
        Add scaffolds to the data frame. A new column is created that contains the SMILES of the corresponding scaffold.
        If `add_rdkit_scaffold` is set to `True`, a new column is created that contains the RDKit scaffold of the
        corresponding molecule.

        Args:
            scaffolds (list): List of `Scaffold` calculators.
            add_rdkit_scaffold (bool): Whether to add the RDKit scaffold of the molecule as a new column.
        """

        for scaffold in scaffolds:
            if f"Scaffold_{scaffold}" in self.df.columns:
                continue

            self.df[f"Scaffold_{scaffold}"] = self.apply(self._scaffold_calculator, func_args=(scaffold,), subset=[self.smilescol], axis=1, raw=True)
            if add_rdkit_scaffold:
                PandasTools.AddMoleculeColumnToFrame(self.df, smilesCol=f"Scaffold_{scaffold}",
                                                 molCol=f"Scaffold_{scaffold}_RDMol")

    def getScaffoldNames(self, include_mols=False):
        """
        Get the names of the scaffolds in the data frame.

        Args:
            include_mols (bool): Whether to include the RDKit scaffold columns as well.
        """

        return [col for col in self.df.columns if
                col.startswith("Scaffold_") and (include_mols or not col.endswith("_RDMol"))]

    def getScaffolds(self, includeMols=False):
        """
        Get the subset of the data frame that contains only scaffolds.

        Args:
            includeMols (bool): Whether to include the RDKit scaffold columns as well.
        """

        if includeMols:
            return self.df[[col for col in self.df.columns if col.startswith("Scaffold_")]]
        else:
            return self.df[self.getScaffoldNames()]

    @property
    def hasScaffolds(self):
        """
        Check whether the data frame contains scaffolds.

        Returns:
            bool: Whether the data frame contains scaffolds.
        """

        return len(self.getScaffoldNames()) > 0

    def createScaffoldGroups(self, mols_per_group=10):
        """
        Create scaffold groups. A scaffold group is a list of molecules that share the same scaffold. New columns are
        created that contain the scaffold group ID and the scaffold group size.

        Args:
            mols_per_group (int): Number of molecules per scaffold group.
        """

        scaffolds = self.getScaffolds(includeMols=False)
        for scaffold in scaffolds.columns:
            counts = pd.value_counts(self.df[scaffold])
            mask = counts.lt(mols_per_group)
            name = f'ScaffoldGroup_{scaffold}_{mols_per_group}'
            if name not in self.df.columns:
                self.df[name] = np.where(self.df[scaffold].isin(counts[mask].index), 'Other',
                                            self.df[scaffold])

    def getScaffoldGroups(self, scaffold_name: str, mol_per_group: int = 10):
        """
        Get the scaffold groups for a given combination of scaffold and number of molecules per scaffold group.

        Args:
            scaffold_name (str): Name of the scaffold.
            mol_per_group (int): Number of molecules per scaffold group.

        Returns:
            list: List of scaffold groups.
        """

        return self.df[
            self.df.columns[self.df.columns.str.startswith(f"ScaffoldGroup_{scaffold_name}_{mol_per_group}")][0]]

    @property
    def hasScaffoldGroups(self):
        """
        Check whether the data frame contains scaffold groups.

        Returns:
            bool: Whether the data frame contains scaffold groups.
        """
        return len([col for col in self.df.columns if col.startswith("ScaffoldGroup_")]) > 0


class QSPRDataset(MoleculeTable):
    """Prepare dataset for QSPR model training.

    It splits the data in train and test set, as well as creating cross-validation folds.
    Optionally low quality data is filtered out.
    For classification the dataset samples are labelled as active/inactive.

    Attributes:
        targetProperty (str) : property to be predicted with QSPRmodel
        task (ModelTask) : regression or classification
        df (pd.dataframe) : dataset
        X (np.ndarray/pd.DataFrame) : m x n feature matrix for cross validation, where m is
            the number of samplesand n is the number of features.
        y (np.ndarray/pd.DataFrame) : m-d label array for cross validation, where m is the
            number of samples and equals to row of X.
        X_ind (np.ndarray/pd.DataFrame) : m x n Feature matrix for independent set, where m
            is the number of samples and n is the number of features.
        y_ind (np.ndarray/pd.DataFrame) : m-l label array for independent set, where m is
            the number of samples and equals to row of X_ind, and l is the number of types.
        n_folds (int) : number of folds for the generator
        features (list of str) : feature names
        feature_standardizers (list of FeatureStandardizers): methods used to standardize the data features.
    """

    def __init__(
        self,
        name: str,
        target_prop: str,
        df: pd.DataFrame = None,
        smilescol: str = "SMILES",
        add_rdkit: bool = False,
        store_dir: str = '.',
        overwrite: bool = False,
        task: Literal[ModelTasks.REGRESSION, ModelTasks.CLASSIFICATION] = ModelTasks.REGRESSION,
        target_transformer: Callable = None,
        th: List[float] = None,
        n_folds=None,
        n_jobs: int = 0,
        chunk_size: int = 50,
    ):
        """Construct QSPRdata, also apply transformations of output property if specified.

        Args:
            name (str): data name, used in saving the data
            target_prop (str): target property, should correspond with target columnname in df
            df (pd.DataFrame, optional): input dataframe containing smiles and target property. Defaults to None.
            smilescol (str, optional): name of column in df containing SMILES. Defaults to "SMILES".
            add_rdkit (bool, optional): if true, column with rdkit molecules will be added to df. Defaults to False.
            store_dir (str, optional): directory for saving the output data. Defaults to '.'.
            overwrite (bool, optional): if already saved data at output dir if should be overwritten. Defaults to False.
            task (Literal[ModelTasks.REGRESSION, ModelTasks.CLASSIFICATION], optional): Defaults to ModelTasks.REGRESSION.
            target_transformer (Callable, optional): Transformation(s) of target propery. Defaults to None.
            th (List[float], optional): threshold for activity if classification model, if len th
                larger than 1, these values will used for binning (in this case lower and upper
                boundary need to be included). Defaults to None.
            n_folds (str): Overwritten in prepare_dataset. This is here for re-loading the model.

        Raises:
            ValueError: Raised if thershold given with non-classification task.
        """
        super().__init__(name, df, smilescol, add_rdkit, store_dir, overwrite, n_jobs, chunk_size)
        self.targetProperty = target_prop
        self.task = task

        self.dropInvalids()

        if target_transformer:
            transformed_prop = f'{self.targetProperty}_transformed'
            self.transform([self.targetProperty], [target_transformer], addAs=[transformed_prop])
            self.targetProperty = f'{self.targetProperty}_transformed'

        if self.task == ModelTasks.CLASSIFICATION:
            if th:
                self.makeClassification(th, as_new=True)
            else:
                # if a precomputed target is expected, just check it
                assert all(float(x).is_integer() for x in self.df[self.targetProperty]), f"Target property ({self.targetProperty}) should be integer if used for classification. Or specify threshold for binning."
        elif self.task == ModelTasks.REGRESSION and th:
            raise ValueError(
                f"Got regression task with specified thresholds: 'th={th}'. Use 'task=ModelType.CLASSIFICATION' in this case.")

        self.X = None
        self.y = None
        self.X_ind = None
        self.y_ind = None

        self.n_folds = n_folds

        self.features = None
        self.feature_standardizers = []
        self.feature_filters = []

        logger.info(f"Dataset '{self.name}' created for target targetProperty: '{self.targetProperty}'.")

    def isMultiClass(self):
        """Return if model task is multi class classification."""
        return self.task == ModelTasks.CLASSIFICATION and self.nClasses > 2

    @property
    def nClasses(self):
        """Return number of output classes for classification."""
        if self.task == ModelTasks.CLASSIFICATION:
            return len(self.df[self.targetProperty].unique())
        else:
            return 0

    def dropInvalids(self):
        """Drop Invalid SMILES and missing target property values."""
        # drop rows with empty targetProperty values
        self.df = self.df.dropna(subset=([self.smilescol, self.targetProperty])).copy()

        # drop invalid smiles
        invalid_mask = self.df[self.smilescol].apply(lambda smile: Chem.MolFromSmiles(smile) is not None)
        logger.info(
            f"Removing invalid SMILES: {self.df[self.smilescol][invalid_mask]}"
        )
        self.df = self.df[invalid_mask].copy()

    def cleanMolecules(self, standardize: bool = True, sanitize: bool = True):
        """Standardize and or sanitize SMILES sequences."""
        if standardize:
            self.standardize()
        if sanitize:
            self.sanitize()

    def standardize(self):
        self.df[self.smilescol] = [chembl_smi_standardizer(smiles)[0] for smiles in self.df[self.smilescol]]

    def sanitize(self):
        self.df[self.smilescol] = [sanitize_smiles(smiles) for smiles in self.df[self.smilescol]]

    def makeClassification(self, th: List[float] = tuple(), as_new: bool = False):
        """Convert model output to classification using the given threshold(s)."""
        new_prop = self.targetProperty if not as_new else f"{self.targetProperty}_class"
        assert len(th) > 0, "Threshold list must contain at least one value."
        if len(th) > 1:
            assert (
                len(th) > 3
            ), "For multi-class classification, set more than 3 values as threshold."
            assert max(self.df[self.targetProperty]) <= max(
                th
            ), "Make sure final threshold value is not smaller than largest value of property"
            assert min(self.df[self.targetProperty]) >= min(
                th
            ), "Make sure first threshold value is not larger than smallest value of property"
            self.df[f"{new_prop}_intervals"] = pd.cut(
                self.df[self.targetProperty], bins=th, include_lowest=True
            ).astype(str)
            self.df[new_prop] = LabelEncoder().fit_transform(self.df[f"{new_prop}_intervals"])
        else:
            self.df[new_prop] = (self.df[self.targetProperty] > th[0]).astype(float)
        self.task = ModelTasks.CLASSIFICATION
        self.targetProperty = new_prop
        self.th = th
        logger.info("Target property converted to classification.")

    @staticmethod
    def fromFile(filename, *args, **kwargs) -> 'QSPRDataset':
        store_dir = os.path.dirname(filename)
        name = os.path.basename(filename).rsplit('_',1)[0]
        with open(os.path.join(store_dir, f"{name}_meta.json")) as f:
            meta = json.load(f)
            meta_init = meta['init']
            meta_init['task'] = ModelTasks(meta_init['task'])

        return QSPRDataset(*args, name=name, store_dir=store_dir, **meta_init, **kwargs)

    @staticmethod
    def fromMolTable(mol_table: MoleculeTable, target_prop : str, name=None, **kwargs) -> 'QSPRDataset':
        """Create QSPRDataset from a MoleculeTable."""
        kwargs['store_dir'] = mol_table.storeDir if 'store_dir' not in kwargs else kwargs['store_dir']
        name = mol_table.name if name is None else name
        return QSPRDataset(name, target_prop, mol_table.getDF(), **kwargs)

    def save(self, save_split=True):
        super().save()

        # save feature standardizers
        stds = []
        for idx, standardizer in enumerate(self.feature_standardizers):
            path = f'{self.storePrefix}_feature_standardizer_{idx}.json'
            if not hasattr(standardizer, 'toFile'):
                SKLearnStandardizer(standardizer).toFile(path)
            else:
                standardizer.toFile(path)
            stds.append(path)

        # save X and y
        if save_split:
            if self.X is not None:
                self.X.to_pickle(f'{self.storePrefix}_X.pkl')
            if self.X_ind is not None:
                self.X_ind.to_pickle(f'{self.storePrefix}_X_ind.pkl')
            if self.y is not None:
                self.y.to_pickle(f'{self.storePrefix}_y.pkl')
            if self.y_ind is not None:
                self.y_ind.to_pickle(f'{self.storePrefix}_y_ind.pkl')

        # save metadata
        meta_init = {
            'target_prop': self.targetProperty,
            'task': self.task.name,
            'n_folds': self.n_folds,
            'smilescol': self.smilescol
        }
        meta_data = {}
        if self.task == ModelTasks.CLASSIFICATION:
            meta_data.update({'th': self.th})
        meta = {
            'init': meta_init,
            'data': meta_data
        }
        with open(f"{self.storePrefix}_meta.json", 'w') as f:
            json.dump(meta, f)

    def reload(self, load_split=True):
        super().reload()
        if load_split:
            try:
                self.X = pd.read_pickle(f"{self.storePrefix}_X.pkl")
                self.X_ind = pd.read_pickle(f"{self.storePrefix}_X_ind.pkl")
                self.y = pd.read_pickle(f"{self.storePrefix}_y.pkl")
                self.y_ind = pd.read_pickle(f"{self.storePrefix}_y_ind.pkl")
            except FileNotFoundError:
                logger.warning("No input or output train/test dataframes saved.")

    def split(self, split: datasplit):
        """Split dataset into train and test set.

        Args:
            split (datasplit) : split instance orchestrating the split
        """
        self.X, self.X_ind, self.y, self.y_ind = split(
            df=self.df, Xcol=self.smilescol, ycol=self.targetProperty
        )
        logger.info("Total: train: %s test: %s" % (len(self.y), len(self.y_ind)))
        if self.task == ModelTasks.CLASSIFICATION:
            if not self.isMultiClass():
                logger.info(
                    "    In train: active: %s not active: %s"
                    % (sum(self.y), len(self.y) - sum(self.y))
                )
                logger.info(
                    "    In test:  active: %s not active: %s\n"
                    % (sum(self.y_ind), len(self.y_ind) - sum(self.y_ind))
                )
            else:
                logger.info("train: %s" % self.y.value_counts())
                logger.info("test: %s\n" % self.y_ind.value_counts())
                try:
                    assert np.all([x > 0 for x in self.y.value_counts()])
                    assert np.all([x > 0 for x in self.y_ind.value_counts()])
                except AssertionError as err:
                    logger.exception(
                        "All bins in multi-class classification should contain at least one sample"
                    )
                    raise err

            if self.y.dtype.name == "category":
                self.y = self.y.cat.codes
                self.y_ind = self.y_ind.cat.codes

    def featurizeSplits(self):
        """Keep only features that will be used by the model. In our case, descriptors."""
        descriptors = self.getDescriptors()
        self.X = descriptors.loc[self.X.index, :]
        self.X_ind = descriptors.loc[self.X_ind.index, :]
        self.y = self.df.loc[self.y.index, [self.targetProperty]]
        self.y_ind = self.df.loc[self.y_ind.index, [self.targetProperty]]

    def fillMissing(self, fill_value: float, columns: List[str] = None):
        columns = columns if columns else self.getDescriptorNames()
        self.df[columns] = self.df[columns].fillna(fill_value)
        logger.warning('Missing values filled with %s' % fill_value)

    def filterFeatures(self, feature_filters=None):
        if feature_filters is not None:
            self.feature_filters = feature_filters
        for featurefilter in self.feature_filters:
            self.X = featurefilter(self.X, self.y)

        self.features = self.X.columns
        self.X_ind = self.X_ind[self.features]
        logger.info(f"Selected features: {self.features}")

        # update descriptor calculator
        self.descriptorCalculator.keepDescriptors(self.features)

    def prepareDataset(
        self,
        standardize=True,
        sanitize=True,
        datafilters=None,
        split=randomsplit(),
        feature_calculator=None,
        feature_filters=None,
        feature_standardizers=None,
        n_folds=5,
        recalculate_features=False,
        fill_value=0
    ):
        """Prepare the dataset for use in QSPR model.

        Arguments:
            standardize (bool): Apply Chembl standardization pipeline to smiles
            sanitize (bool): sanitize smiles
            datafilters (list of datafilter obj): filters number of rows from dataset
            split (datasplitter obj): splits the dataset into train and test set
            feature_calculator (DescriptorsCalculator): calculates features from smiles
            feature_filters (list of feature filter objs): filters features
            feature_standardizers (list of feature standardizer objs): standardizes and/or scales features
            n_folds (int): number of folds to use in cross-validation
            recalculate_features (bool): recalculate features even if they are already present in the file
            fill_value (float): value to fill missing values with
        """
        # apply sanitization and standardization
        if standardize:
            self.cleanMolecules(standardize, sanitize)

        # calculate features
        if feature_calculator is not None:
            self.addDescriptors(feature_calculator, recalculate=recalculate_features)

        # apply data filters
        if datafilters is not None:
            self.filter(datafilters)

        # Replace any NaN values in features by 0
        # FIXME: this is not very good, we should probably add option to do data imputation here or drop rows with NaNs
        if fill_value is not None:
            self.fillMissing(fill_value)

        # split dataset
        if split is not None:
            self.split(split)

        # featurize splits
        if self.hasDescriptors:
            self.featurizeSplits()

        # apply feature filters on training set
        if feature_filters is not None:
            if not self.hasDescriptors:
                logger.warning(
                    "No descriptors present, feature filters will be applied, but might have no effect."
                )
            self.filterFeatures(feature_filters)

        # standardize features in the main data set
        if feature_standardizers is not None:
            if not self.hasDescriptors:
                logger.warning(
                    "No descriptors present, feature standardizers will be applied, but might have no effect."
                )
            self.standardizeFeatures(feature_standardizers)

        # create folds for cross-validation
        self.createFolds(n_folds=n_folds)

    @staticmethod
    def applyFeatureStandardizers(feature_standardizers, X, fit=True):
        """Apply and/or fit feature standardizers."""
        fitted_standardizers = []
        for idx, standardizer in enumerate(feature_standardizers):
            if isinstance(standardizer, SKLearnStandardizer):
                standardizer = standardizer.getInstance()

            if fit:
                standardizer = SKLearnStandardizer.fromFit(X, standardizer)
            else:
                standardizer = SKLearnStandardizer(standardizer)

            X = standardizer(X)
            fitted_standardizers.append(standardizer)

        return X, fitted_standardizers

    def createFolds(self, n_folds=None, feature_standardizers=tuple()):
        """Create folds for crossvalidation."""
        self.n_folds = n_folds if n_folds else self.n_folds
        if not self.n_folds:
            raise ValueError("Number of folds not specified nor in class property or as an argument.")
        feature_standardizers = feature_standardizers if feature_standardizers else self.feature_standardizers if self.feature_standardizers else tuple()
        if self.task != ModelTasks.CLASSIFICATION:
            folds = KFold(self.n_folds).split(self.X)
        else:
            folds = StratifiedKFold(self.n_folds).split(self.X, self.y)

        def standardize_folds(folds):
            for x in folds:
                X, standardizers = self.applyFeatureStandardizers(
                    feature_standardizers, self.X.values[x[0], :], fit=True)
                X_test, _ = self.applyFeatureStandardizers(standardizers, self.X.values[x[1], :], fit=False)
                y = self.y.values[x[0]]
                y_test = self.y.values[x[1]]
                yield X, X_test, y[:, 0], y_test[:, 0], x[0], x[1]

        if hasattr(self, "feature_standardizers"):
            folds = standardize_folds(folds)
        logger.debug("Folds created for crossvalidation")

        return folds

    def standardizeFeatures(self, feature_standardizers=None):
        if feature_standardizers is not None:
            self.feature_standardizers = feature_standardizers
        if not self.feature_standardizers:
            raise ValueError("No feature standardizers specified in class or in argument.")
        X, self.feature_standardizers = self.applyFeatureStandardizers(
            self.feature_standardizers,
            self.X.values,
            fit=True
        )
        X_ind, _ = self.applyFeatureStandardizers(
            self.feature_standardizers,
            self.X_ind.values,
            fit=False
        )

        self.X = pd.DataFrame(X, index=self.X.index, columns=self.X.columns)
        self.X_ind = pd.DataFrame(X_ind, index=self.X_ind.index, columns=self.X_ind.columns)
