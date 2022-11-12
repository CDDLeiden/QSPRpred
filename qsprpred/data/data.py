"""This module contains the QSPRDataset that holds and prepares data for modelling."""
import numpy as np
import pandas as pd
from qsprpred.data.utils.datasplitters import randomsplit
from qsprpred.data.utils.featurefilters import BorutaFilter
from qsprpred.data.utils.smiles_standardization import (
    chembl_smi_standardizer,
    sanitize_smiles,
)
from qsprpred.logs import logger
from rdkit import Chem
from rdkit.Chem import PandasTools
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler as Scaler


class QSPRDataset:
    """Prepare dataset for QSPR model training.
    
    It splits the data in train and test set, as well as creating cross-validation folds.
    Optionally low quality data is filtered out.
    For classification the dataset samples are labelled as active/inactive.

    Attributes:
    df (pd dataframe) : dataset
    smilescol (str) : name of column containing the molecule smiles
    property (str) : name of column in dataframe for to be predicted values, e.g. ["Cl"]
    reg (bool) : if true, dataset for regression, if false dataset for classification
        (uses th)
    th (list of float) : threshold for activity if classification model, if len th
        larger than 1, these values will used for binning (in this case lower and upper
        boundary need to be included)
    X (np.ndarray/pd.DataFrame) : m x n feature matrix for cross validation, where m is
        the number of samplesand n is the number of features.
    y (np.ndarray/pd.DataFrame) : m-d label array for cross validation, where m is the
        number of samples and equals to row of X.
    X_ind (np.ndarray/pd.DataFrame) : m x n Feature matrix for independent set, where m
        is the number of samples and n is the number of features.
    y_ind (np.ndarray/pd.DataFrame) : m-l label array for independent set, where m is 
        the number of samples and equals to row of X_ind, and l is the number of types.
    folds (generator) : scikit-learn n-fold generator object
    n_folds (int) : number of folds for the generator

    Methods:
    FromFile : construct dataset from file
    prepareDataset : preprocess the dataset for QSPR modelling
    loadFeaturesFromFile: load features from file :)
    createFolds: folds is an generator and needs to be reset after cross validation or hyperparameter optimization
    dataStandardization: Performs standardization by centering and scaling
    """

    def __init__(
        self, df: pd.DataFrame, property, smilescol="SMILES", reg=True, th=[], log=False
    ):
        self.smilescol = smilescol
        self.property = property
        self.df = df.dropna(subset=([smilescol, property])).copy()

        # drop invalid smiles
        PandasTools.AddMoleculeColumnToFrame(
            self.df, smilescol, "Mol", includeFingerprints=False
        )
        logger.info(
            f"Removed invalid Smiles: {self.df.iloc[np.where(self.df['Mol'].isnull())[0]][smilescol]}"
        )
        self.df = self.df.dropna(subset=(["Mol"]))

        self.reg = reg

        if reg and log:
            self.df[property] = np.log(self.df[property])

        self.th = [] if reg else th
        if not reg:
            assert type(th) == list, "thresholds should be a list"
            if len(th) > 1:
                assert (
                    len(th) > 3
                ), "For multi-class classification, set more than 3 values as threshold."
                assert max(self.df[property]) <= max(
                    self.th
                ), "Make sure final threshold value is not smaller than largest value of property"
                assert min(self.df[property]) >= min(
                    self.th
                ), "Make sure first threshold value is not larger than smallest value of property"
                self.df[property] = pd.cut(
                    self.df[property], bins=th, include_lowest=True
                )
            else:
                self.df[property] = (self.df[property] > self.th[0]).astype(float)

        self.X = None
        self.y = None
        self.X_ind = None
        self.y_ind = None

        self.n_folds = None
        self.folds = None

        self.features = None
        self.target_desc = None

        logger.info(f"Dataset created for {property}")

    @classmethod
    def FromFile(cls, fname, *args, **kwargs):
        df = pd.read_csv(fname, sep="\t")
        return QSPRDataset(df, *args, **kwargs)

    def prepareDataset(
        self,
        fname,
        standardize=True,
        sanitize=True,
        datafilters=[],
        split=randomsplit(),
        feature_calculators=None,
        featurefilters=[],
        n_folds=5,
    ):
        """Prepare the dataset for use in QSPR model.

        Arguments:
            fname (str): feature_calculator with filtered features saved to this file
            standarize (bool): Apply Chembl standardization pipeline to smiles
            sanitize (bool): sanitize smiles
            datafilters (list of datafilter obj): filters number of rows from dataset
            split (datasplitter obj): splits the dataset into train and test set
            features (list of feature calculation cls): calculates features from smiles
            featurefilters (list of feature filter objs): filters features
            n_folds (n): number of folds to use in cross-validation
        """
        # standardize and sanitize smiles
        if standardize:
            self.df[self.smilescol] = [chembl_smi_standardizer(smiles)[0] for smiles in self.df[self.smilescol]]
        if sanitize:
            self.df[self.smilescol] = [sanitize_smiles(smiles) for smiles in self.df[self.smilescol]]

        # apply filters on dataset
        for filter in datafilters:
            self.df = filter(self.df)

        # split dataset in train and test set
        self.X, self.X_ind, self.y, self.y_ind = split(
            df=self.df, Xcol=self.smilescol, ycol=self.property
        )
        logger.info("Total: train: %s test: %s" % (len(self.y), len(self.y_ind)))
        if not self.reg:
            if len(self.th) == 1:
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

        # calculate features from smiles
        self.X = feature_calculators(
            [Chem.MolFromSmiles(mol) for mol in self.X]
        )
        self.X_ind = feature_calculators(
            [Chem.MolFromSmiles(mol) for mol in self.X_ind]
        )

        # Replace any NaN values in features by 0
        self.X = self.X.fillna(0)
        self.X_ind = self.X.fillna(0)
        
        # apply filters to features on trainingset
        for featurefilter in featurefilters:
            if type(featurefilter) == BorutaFilter:
                self.X = featurefilter(self.X, self.y)
            else:
                self.X = featurefilter(self.X)

        self.features = self.X.columns
        self.X_ind = self.X_ind[self.features]
        logger.info(f"Selected features: {self.features}")

        # drop removed from feature_calulator object
        for idx, descriptorset in enumerate(feature_calculators.descsets):
            descs_from_curr_set = [
                f.removeprefix(f"{descriptorset}_")
                for f in self.features
                if f.startswith(str(descriptorset))
            ]
            if not descs_from_curr_set:
                feature_calculators.descsets.remove(descriptorset)
            elif descriptorset.is_fp:
                feature_calculators.descsets[idx].keepindices = [
                    f for f in descs_from_curr_set
                ]
            else:
                feature_calculators.descsets[idx].descriptors = descs_from_curr_set

        feature_calculators.toFile(fname)

        self.X = np.array(self.X)
        self.X_ind = np.array(self.X_ind)
        self.y = np.array(self.y)
        self.y_ind = np.array(self.y_ind)

        # create folds for cross-validation
        self.n_folds = n_folds
        self.createFolds()

    def loadFeaturesFromFile(fname: str) -> None:
        """Load in calculated features from file.
        
        Useful if features were calculated in other software, such as MOE.
        Features are added to X and X_ind
        
        Arguments:
            fname (str): file name of feature file
        """
        # TODO implement this function
        pass

    def createFolds(self):
        """Create folds for crossvalidation."""
        if self.reg:
            self.folds = KFold(self.n_folds).split(self.X)
        else:
            self.folds = StratifiedKFold(self.n_folds).split(self.X, self.y)
        logger.debug("Folds created for crossvalidation")

    @staticmethod
    def dataStandardization(data_x, test_x):
        """Perform standardization by centering and scaling.

        Arguments:
            data_x (list): descriptors of data set
            test_x (list): descriptors of test set

        Returns:
            data_x (list): descriptors of data set standardized
            test_x (list): descriptors of test set standardized
        """
        scaler = Scaler()
        scaler.fit(data_x)
        test_x = scaler.transform(test_x)
        data_x = scaler.transform(data_x)
        logger.debug("Data standardized")
        return data_x, test_x
