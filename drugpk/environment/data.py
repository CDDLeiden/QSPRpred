from drugpk.training.scorers.predictors import Predictor
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.model_selection import StratifiedKFold, KFold
from drugpk.logs import logger
from drugpk.environment.dataprep_utils.datasplitters import randomsplit
import pandas as pd
import numpy as np

class QSKRDataset:
    """
        This class is used to prepare the dataset for QSKR model training. 
        It splits the data in train and test set, as well as creating cross-validation folds.
        Optionally low quality data is filtered out.
        For classification the dataset samples are labelled as active/inactive.
        ...

        Attributes
        ----------
        df (pd dataframe)               : dataset
        smilescol (str)                 : name of column containing the molecule smiles
        properties (list of str)        : name of column(s) in dataframe for to be predicted values, e.g. ["Cl"]
        reg (bool)                      : if true, dataset for regression, if false dataset for classification (uses th)
        th (list of float)              : threshold for activity if classficiation model, ignored otherwise

        X (np.ndarray/pd.DataFrame)     : m x n feature matrix for cross validation, where m is the number of samples
                                          and n is the number of features.
        y (np.ndarray/pd.DataFrame)     : m-d label array for cross validation, where m is the number of samples and
                                          equals to row of X.
        X_ind (np.ndarray/pd.DataFrame) : m x n Feature matrix for independent set, where m is the number of samples
                                          and n is the number of features.
        y_ind (np.ndarray/pd.DataFrame) : m-l label array for independent set, where m is the number of samples and
                                          equals to row of X_ind, and l is the number of types.
        folds (generator)               : scikit-learn n-fold generator object
        n_folds (int)                   : number of folds for the generator

        Methods
        -------
        splitDataset : A train and test split is made.
        createFolds: folds is an generator and needs to be reset after cross validation or hyperparameter optimization
        dataStandardization: Performs standardization by centering and scaling
    """

    def __init__(self, df: pd.DataFrame, smilescol = 'SMILES', properties = ['CL'], reg=True, th=[6.5]):
        assert type(properties) == list, "properties should be a list"
        
        self.smilescol = smilescol
        self.properties = properties
        self.df = df.dropna(subset=([smilescol] + properties))
        
        self.reg = reg

        if not reg:
            self.th = th
            assert type(th) == list, "thresholds should be a list"
            if len(th) > 1:
                for prop in properties:
                    df[prop] = pd.cut(df[prop], bins=th)
            else:
                df[prop] = (df[prop] > self.th).astype(float)

        self.X = None
        self.y = None
        self.X_ind = None
        self.y_ind = None
        self.n_folds = None
        self.folds = None

    @classmethod
    def FromFile(cls, fname, smilescol = 'SMILES', property = 'CL', reg=True, th=6.5):
        df = pd.read_csv(fname, sep="\t")
        return QSKRDataset(df, smilescol, property, reg, th)

    def prepareDataset(self, datafilters=[], split=randomsplit(), features=[Predictor.calculateDescriptors],
                       featurefilters=[], n_folds=5):
        """
            prepare the dataset for use in QSKR model
            Arguments:
                datafilters (list of datafilter obj): filters number of rows from dataset
                split (datasplitter obj): splits the dataset into train and test set
                features (list of feature calculation cls): calculates features from smiles
                featurefilters (list of feature filter objs): filters features
                n_folds (n): number of folds to use in cross-validation
        """

        # apply filters on dataset    
        for filter in datafilters:
            self.input_df = filter(self.input_df)

        # split dataset in train and test set
        self.X, self.X_ind, self.y, self.y_ind = split(self.input_df)

        # calculate features from smiles
        self.X = features([Chem.MolFromSmiles(mol) for mol in self.X])
        self.Xind = features([Chem.MolFromSmiles(mol) for mol in self.Xind])

        # apply filters to features
        alldata = np.concatenate([self.X, self.Xind], axis=0)
        for featurefilter in featurefilters:
            selected_features = featurefilter(alldata)
            logger.info(f"{alldata.shape[1] - len(selected_features)} dropped from calculated features by {type(featurefilter)}")
            self.X = self.X[selected_features]
            self.Xind = self.Xind

        # create folds for cross-validation
        self.n_folds = n_folds
        self.createFolds()

        #Write information about the trainingset to the logger
        logger.info('Train and test set created for %s %s:' % (self.valuecol, 'REG' if self.reg else 'CLS'))
        logger.info('    Total: train: %s test: %s' % (len(data), len(test)))
        if self.reg:
            logger.info('    Total: active: %s not active: %s' % (sum(self.df >= self.th), sum(self.df < self.th)))
            logger.info('    In train: active: %s not active: %s' % (sum(data >= self.th), sum(data < self.th)))
            logger.info('    In test:  active: %s not active: %s\n' % (sum(test >= self.th), sum(test < self.th)))
        else:
            logger.info('    Total: active: %s not active: %s' % (self.df.sum().astype(int), (len(self.df)-self.df.sum()).astype(int)))
            logger.info('    In train: active: %s not active: %s' % (data.sum().astype(int), (len(data)-data.sum()).astype(int)))
            logger.info('    In test:  active: %s not active: %s\n' % (test.sum().astype(int), (len(test)-test.sum()).astype(int)))
    
    def loadFeaturesFromFile(fname: str) -> None:
        """
            Function to load in calculated features from file
            Useful if features were calculated in other software, such as MOE.
            Features are added to X and X_ind
            Arguments:
                fname (str): file name of feature file
        """
        #TODO implement this function
        pass

    def createFolds(self):
        """
            Create folds for crossvalidation
        """
        if self.reg:
            self.folds = KFold(self.n_folds).split(self.X)
        else:
            self.folds = StratifiedKFold(self.n_folds).split(self.X, self.y)
        logger.debug("Folds created for crossvalidation")
        
    @staticmethod
    def dataStandardization(data_x, test_x):
        """
        Perform standardization by centering and scaling

        Arguments:
                    data_x (list): descriptors of data set
                    test_x (list): descriptors of test set
        
        Returns:
                    data_x (list): descriptors of data set standardized
                    test_x (list): descriptors of test set standardized
        """
        scaler = Scaler(); scaler.fit(data_x)
        test_x = scaler.transform(test_x)
        data_x = scaler.transform(data_x)
        logger.debug("Data standardized")
        return data_x, test_x