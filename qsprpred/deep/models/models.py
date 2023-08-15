"""Here the DNN model originally from DrugEx can be found.

At the moment this contains a class for fully-connected DNNs.
To add more a model class implementing the `QSPRModel` interface can be added,
see tutorial adding_new_components.
"""
import os
from typing import Any, Type

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from ...data.data import QSPRDataset
from ...deep import DEFAULT_DEVICE, DEFAULT_GPUS, SSPACE
from ...deep.models.neural_network import STFullyConnected
from ...models.interfaces import QSPRModel
from ...models.tasks import ModelTasks
from ...models.early_stopping import EarlyStoppingMode, early_stopping
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange
from copy import deepcopy
import shutil
import chemprop
from ...deep.models.chemprop_wrapper import MoleculeModel


class QSPRDNN(QSPRModel):
    """This class holds the methods for training and fitting a
    Deep Neural Net QSPR model initialization.

    Here the model instance is created and parameters can be defined.

    Attributes:
        name (str): name of the model
        data (QSPRDataset): data set used to train the model
        alg (estimator): estimator instance or class
        parameters (dict): dictionary of algorithm specific parameters
        estimator (object):
            the underlying estimator instance, if `fit` or optimization is perforemed,
            this model instance gets updated accordingly
        featureCalculators (MoleculeDescriptorsCalculator):
            feature calculator instance taken from the data set
            or deserialized from file if the model is loaded without data
        featureStandardizer (SKLearnStandardizer):
            feature standardizer instance taken from the data set
            or deserialized from file if the model is loaded without data
        metaInfo (dict):
            dictionary of metadata about the model, only available
            after the model is saved
        baseDir (str):
            base directory of the model, the model files
            are stored in a subdirectory `{baseDir}/{outDir}/`
        metaFile (str):
            absolute path to the metadata file of the model (`{outPrefix}_meta.json`)
        device (cuda device): cuda device
        gpus (int/ list of ints): gpu number(s) to use for model fitting
        patience (int):
            number of epochs to wait before early stop if no progress
            on validation set score
        tol (float):
            minimum absolute improvement of loss necessary to count as
            progress on best validation score
        nClass (int): number of classes
        nDim (int): number of features
        optimalEpochs (int): number of epochs to train the model for optimal performance
        device (torch.device): cuda device, cpu or gpu
        gpus (list[int]): gpu number(s) to use for model fitting
        patience (int):
            number of epochs to wait before early stop
            if no progress on validation set score
        optimalEpochs (int):
            number of epochs to train the model for optimal performance
    """
    def __init__(
        self,
        base_dir: str,
        alg: Type = STFullyConnected,
        data: QSPRDataset | None = None,
        name: str | None = None,
        parameters: dict | None = None,
        autoload: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        gpus: list[int] = DEFAULT_GPUS,
        patience: int = 50,
        tol: float = 0,
    ):
        """Initialize a QSPRDNN model.

        Args:
            base_dir (str):
                base directory of the model, the model files are stored in
                a subdirectory `{baseDir}/{outDir}/`
            alg (Type, optional):
                model class or instance. Defaults to STFullyConnected.
            data (QSPRDataset, optional):
                data set used to train the model. Defaults to None.
            name (str, optional):
                name of the model. Defaults to None.
            parameters (dict, optional):
                dictionary of algorithm specific parameters. Defaults to None.
            autoload (bool, optional):
                whether to load the model from file or not. Defaults to True.
            device (torch.device, optional):
                The cuda device. Defaults to `DEFAULT_DEVICE`.
            gpus (list[int], optional):
                gpu number(s) to use for model fitting. Defaults to `DEFAULT_GPUS`.
            patience (int, optional):
                number of epochs to wait before early stop if no progress
                on validation set score. Defaults to 50.
            tol (float, optional):
                minimum absolute improvement of loss necessary to count as progress
                on best validation score. Defaults to 0.
        """
        self.device = device
        self.gpus = gpus
        self.patience = patience
        self.tol = tol
        self.nClass = None
        self.nDim = None
        super().__init__(base_dir, alg, data, name, parameters, autoload=autoload)
        if self.task.isMultiTask():
            raise NotImplementedError(
                "Multitask modelling is not implemented for QSPRDNN models."
            )

    @property
    def supportsEarlyStopping(self) -> bool:
        """Whether the model supports early stopping or not."""
        return True

    @classmethod
    def getDefaultParamsGrid(cls) -> list[list]:
        return SSPACE

    def loadEstimator(self, params: dict | None = None) -> object:
        """Load model from file or initialize new model.

        Args:
            params (dict, optional): model parameters. Defaults to None.

        Returns:
            model (object): model instance
        """
        if self.task.isRegression():
            self.nClass = 1
        else:
            self.nClass = (
                self.data.targetProperties[0].nClasses
                if self.data else self.metaInfo["n_class"]
            )
        self.nDim = self.data.X.shape[1] if self.data else self.metaInfo["n_dim"]
        # initialize model
        estimator = self.alg(
            n_dim=self.nDim,
            n_class=self.nClass,
            device=self.device,
            gpus=self.gpus,
            is_reg=self.task == ModelTasks.REGRESSION,
            patience=self.patience,
            tol=self.tol,
        )
        # set parameters if available and return
        new_parameters = self.getParameters(params)
        if new_parameters is not None:
            estimator.set_params(**new_parameters)
        return estimator

    def loadEstimatorFromFile(
        self, params: dict | None = None, fallback_load: bool = True
    ) -> object:
        """Load estimator from file.

        Args:
            params (dict): parameters
            fallback_load (bool):
                if `True`, init estimator from `alg` and `params` if no estimator
                found at path

        Returns:
            estimator (object): estimator instance
        """
        path = f"{self.outPrefix}_weights.pkg"
        estimator = self.loadEstimator(params)
        # load states if available
        if os.path.exists(path):
            estimator.load_state_dict(torch.load(path))
        elif not fallback_load:
            raise FileNotFoundError(
                f"No estimator found at {path}, "
                f"loading estimator weights from file failed."
            )
        return estimator

    def saveEstimator(self) -> str:
        """Save the QSPRDNN model.

        Returns:
            str: path to the saved model
        """
        path = f"{self.outPrefix}_weights.pkg"
        torch.save(self.estimator.state_dict(), path)
        return path

    def saveParams(self, params: dict) -> str:
        """Save model parameters to file.

        Args:
            params (dict): model parameters

        Returns:
            str: path to the saved parameters as json
        """
        return super().saveParams(
            {
                k: params[k]
                for k in params
                if not k.startswith("_") and k not in ["training", "device", "gpus"]
            }
        )

    def save(self) -> str:
        """Save the QSPRDNN model and meta information.

        Returns:
            str: path to the saved model
        """
        self.metaInfo["n_dim"] = self.nDim
        self.metaInfo["n_class"] = self.nClass
        return super().save()

    @early_stopping
    def fit(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        y: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any | None = None,
        mode: EarlyStoppingMode = EarlyStoppingMode.NOT_RECORDING,
        **kwargs
    ):
        """Fit the model to the given data matrix or `QSPRDataset`.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to fit
            y (pd.DataFrame, np.ndarray, QSPRDataset): target matrix to fit
            estimator (Any): estimator instance to use for fitting
            mode (EarlyStoppingMode): early stopping mode
            kwargs (dict): additional keyword arguments for the estimator's fit method

        Returns:
            Any: fitted estimator instance
            int, optional: in case of early stopping, the number of iterations
                after which the model stopped training
        """
        estimator = self.estimator if estimator is None else estimator
        X, y = self.convertToNumpy(X, y)

        if self.earlyStopping:
            # split cross validation fold train set into train
            # and validation set for early stopping
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
            return estimator.fit(X_train, y_train, X_val, y_val, **kwargs)

        # set fixed number of epochs if early stopping is not used
        estimator.n_epochs = self.earlyStopping.getEpochs()
        return estimator.fit(X, y, **kwargs)

    def predict(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any = None
    ) -> np.ndarray:
        """See `QSPRModel.predict`."""
        estimator = self.estimator if estimator is None else estimator
        scores = self.predictProba(X, estimator)
        # return class labels for classification
        if self.task.isClassification():
            return np.argmax(scores[0], axis=1, keepdims=True)
        else:
            return scores[0]

    def predictProba(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any = None
    ) -> np.ndarray:
        """See `QSPRModel.predictProba`."""
        estimator = self.estimator if estimator is None else estimator
        X = self.convertToNumpy(X)

        return [estimator.predict(X)]


class Chemprop(QSPRModel):
    """QSPRpred implementation of Chemprop model.

    Attributes:
        name (str): name of the model
        data (QSPRDataset): data set used to train the model
        alg (Type): estimator class
        parameters (dict): dictionary of algorithm specific parameters
        estimator (Any):
            the underlying estimator instance of the type specified in `QSPRModel.alg`,
            if `QSPRModel.fit` or optimization was performed
        featureCalculators (MoleculeDescriptorsCalculator):
            feature calculator instance taken from the data set or
            deserialized from file if the model is loaded without data
        featureStandardizer (SKLearnStandardizer):
            feature standardizer instance taken from the data set
            or deserialized from file if the model is loaded without data
        metaInfo (dict):
            dictionary of metadata about the model,
            only available after the model is saved
        baseDir (str):
            base directory of the model,
            the model files are stored in a subdirectory `{baseDir}/{outDir}/`
        metaFile (str):
            absolute path to the metadata file of the model (`{outPrefix}_meta.json`)
    """
    def __init__(
        self,
        base_dir: str,
        data: QSPRDataset | None = None,
        name: str | None = None,
        parameters: dict | None = None,
        autoload=True,
        quiet_logger: bool = True
    ):
        """Initialize a Chemprop instance.

        If the model is loaded from file, the data set is not required.
        Note that the data set is required for fitting and optimization.

        Args:
            base_dir (str):
                base directory of the model, the model files are stored in a
                subdirectory `{baseDir}/{outDir}/`
            data (QSPRDataset): data set used to train the model
            name (str): name of the model
            parameters (dict): dictionary of algorithm specific parameters
            autoload (bool):
                if `True`, the estimator is loaded from the serialized file
                if it exists, otherwise a new instance of alg is created
            quiet_logger (bool):
                if `True`, the chemprop logger is set to quiet mode (no debug messages)
        """
        alg = MoleculeModel  # wrapper for chemprop.models.MoleculeModel
        self.quietLogger = quiet_logger
        super().__init__(base_dir, alg, data, name, parameters, autoload)
        self.chempropLogger = chemprop.utils.create_logger(
            name="chemprop_logger", save_dir=self.outDir, quiet=quiet_logger
        )

    def supportsEarlyStopping(self) -> bool:
        """Return if the model supports early stopping.

        Returns:
            bool: True if the model supports early stopping
        """
        return True

    @early_stopping
    def fit(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        y: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any = None,
        mode: EarlyStoppingMode = EarlyStoppingMode.NOT_RECORDING,
        keep_logs: bool = False
    ) -> Any | tuple[MoleculeModel, int | None]:
        """Fit the model to the given data matrix or `QSPRDataset`.

        Note. convertToNumpy can be called here, to convert the input data to
        np.ndarray format.

        Note. if no estimator is given, the estimator instance of the model
              is used.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to fit
            y (pd.DataFrame, np.ndarray, QSPRDataset): target matrix to fit
            estimator (Any): estimator instance to use for fitting
            early_stopping (bool): if True, early stopping is used,
                                   only applies to models that support early stopping.

        Returns:
            Any: fitted estimator instance
            int: in case of early stopping, the number of iterations
                after which the model stopped training
        """
        estimator = self.estimator if estimator is None else estimator

        # convert data to chemprop MoleculeDataset
        data = self.convertToMoleculeDataset(X, y)
        args = estimator.args

        # Set pytorch seed for random initial weights
        torch.manual_seed(estimator.args.pytorch_seed)

        # Split data
        debug(f"Splitting data with seed {args.seed}")

        # Set pytorch seed for random initial weights
        torch.manual_seed(args.pytorch_seed)

        # set task names
        args.task_names = [prop.name for prop in self.data.targetProperties]

        # Create validation data when using early stopping
        if self.earlyStopping:
            debug(f"Splitting data with seed {args.seed}")
            train_data, val_data, _ = chemprop.data.utils.split_data(
                data=data,
                split_type=args.split_type,
                sizes=args.split_sizes if args.split_sizes[2] == 0 else [0.9, 0.1, 0],
                seed=args.seed,
                args=args,
                logger=self.chempropLogger
            )
        else:
            train_data = data

        # Get number of molecules per class in training data
        if args.dataset_type == "classification":
            class_sizes = chemprop.data.utils.get_class_sizes(data)
            debug("Class sizes")
            for i, task_class_sizes in enumerate(class_sizes):
                debug(
                    f"{args.task_names[i]} "
                    f"{', '.join(f'{cls}: {size * 100:.2f}%' for cls, size in enumerate(task_class_sizes))}"  # noqa: E501
                )
            train_class_sizes = chemprop.data.utils.get_class_sizes(
                train_data, proportion=False
            )
            args.train_class_sizes = train_class_sizes

        # Get length of training data
        args.train_data_size = len(train_data)

        # log data size
        debug(f"Total size = {len(data):,}")
        if self.earlyStopping:
            debug(f"train size = {len(train_data):,} | val size = {len(val_data):,}")

        # Initialize scaler and standard scale training targets (regression only)
        if args.dataset_type == "regression":
            debug("Fitting scaler")
            estimator.scaler = train_data.normalize_targets()
        else:
            estimator.scaler = None

        # Get loss function
        loss_func = chemprop.train.loss_functions.get_loss_func(args)

        # Automatically determine whether to cache
        if len(data) <= args.cache_cutoff:
            chemprop.data.set_cache_graph(True)
            num_workers = 0
        else:
            chemprop.data.set_cache_graph(False)
            num_workers = args.num_workers

        # Create data loaders
        train_data_loader = chemprop.data.MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=num_workers,
            class_balance=args.class_balance,
            shuffle=True,
            seed=args.seed
        )
        if self.earlyStopping:
            val_data_loader = chemprop.data.MoleculeDataLoader(
                dataset=val_data, batch_size=args.batch_size, num_workers=num_workers
            )

        if args.class_balance:
            debug(
                f"With class_balance, \
                effective train size = {train_data_loader.iter_size:,}"
            )

        # Tensorboard writer
        save_dir = os.path.join(self.outDir, "tensorboard")
        os.makedirs(save_dir, exist_ok=True)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:  # noqa: E722
            writer = SummaryWriter(logdir=save_dir)

        debug(
            f"Number of parameters = {chemprop.nn_utils.param_count_all(estimator):,}"
        )

        if args.cuda:
            debug("Moving model to cuda")
        estimator = estimator.to(args.device)

        # Optimizers
        optimizer = chemprop.utils.build_optimizer(estimator, args)

        # Learning rate schedulers
        scheduler = chemprop.utils.build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float("inf") if args.minimize_score else -float("inf")
        best_epoch, n_iter = 0, 0
        # Get early stopping number of epochs early stopping in case of FIXED or OPTIMAL
        # mode
        n_epochs = self.earlyStopping.getEpochs(
        ) if not self.earlyStopping else args.epochs
        for epoch in trange(n_epochs):
            debug(f"Epoch {epoch}")
            n_iter = chemprop.train.train(
                model=estimator,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=self.chempropLogger,
                writer=writer
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            if self.earlyStopping:
                val_scores = chemprop.train.evaluate(
                    model=estimator,
                    data_loader=val_data_loader,
                    num_tasks=args.num_tasks,
                    metrics=args.metrics,
                    dataset_type=args.dataset_type,
                    scaler=estimator.scaler,
                    logger=self.chempropLogger
                )

                for metric, scores in val_scores.items():
                    # Average validation score\
                    mean_val_score = chemprop.utils.multitask_mean(
                        scores, metric=metric
                    )
                    debug(f"Validation {metric} = {mean_val_score:.6f}")
                    writer.add_scalar(f"validation_{metric}", mean_val_score, n_iter)

                    if args.show_individual_scores:
                        # Individual validation scores
                        for task_name, val_score in zip(args.task_names, scores):
                            debug(f"Validation {task_name} {metric} = {val_score:.6f}")
                            writer.add_scalar(
                                f"validation_{task_name}_{metric}", val_score, n_iter
                            )

                # Save model checkpoint if improved validation score
                mean_val_score = chemprop.utils.multitask_mean(
                    val_scores[args.metric], metric=args.metric
                )
                if args.minimize_score and mean_val_score < best_score or \
                        not args.minimize_score and mean_val_score > best_score:
                    best_score, best_epoch = mean_val_score, epoch
                    best_estimator = deepcopy(estimator)
                # Evaluate on test set using model with best validation score
                info(
                    f"Model best validation {args.metric} = {best_score:.6f} on epoch \
                    {best_epoch}"
                )

        writer.close()
        if not keep_logs:
            # remove temp directory with logs
            shutil.rmtree(save_dir)

        if self.earlyStopping:
            return best_estimator, best_epoch
        return estimator, None

    def predict(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: MoleculeModel | None = None
    ) -> np.ndarray:
        """Make predictions for the given data matrix or `QSPRDataset`.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to predict
            estimator (MoleculeModel): estimator instance to use for fitting

        Returns:
            np.ndarray:
                2D array containing the predictions, where each row corresponds
                to a sample in the data and each column to a target property
        """
        if self.task.isClassification():
            # convert predictions from predictProba to class labels
            preds = self.predictProba(X, estimator)
            preds = [
                np.argmax(preds[i], axis=1, keepdims=True) for i in range(len(preds))
            ]
            # change preds from list of 2D arrays to 2D array
            preds = np.concatenate(preds, axis=1)
            return preds
        return self.predictProba(X, estimator)

    def predictProba(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: MoleculeModel | None = None
    ) -> list[np.ndarray]:
        """Make predictions for the given data matrix or `QSPRDataset`,
        but use probabilities for classification models. Does not work with
        regression models.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to make predict
            estimator (MoleculeModel, None): estimator instance to use for fitting

        Returns:
            list[np.ndarray]:
                a list of 2D arrays containing the probabilities for each class,
                where each array corresponds to a target property, each row
                to a sample in the data and each column to a class
        """
        # Prepare estimator and data
        estimator = self.estimator if estimator is None else estimator
        X = self.convertToMoleculeDataset(X)
        args = estimator.args

        X_loader = chemprop.data.MoleculeDataLoader(
            dataset=X, batch_size=args.batch_size
        )

        # Make predictions
        preds = chemprop.train.predict(
            model=estimator,
            data_loader=X_loader,
            scaler=estimator.scaler,
            disable_progress_bar=True
        )

        # change list of lists to 2D array
        preds = np.array(preds)

        if self.task.isClassification():
            if self.task in [ModelTasks.MULTICLASS, ModelTasks.MULTITASK_MULTICLASS]:
                # chemprop returns 3D array (samples, targets, classes)
                # split into list of 2D arrays (samples, classes), length = n targets
                preds = np.split(preds, preds.shape[1], axis=1)
                preds = [np.squeeze(pred, axis=1) for pred in preds]
                return preds
            elif self.task == ModelTasks.MULTITASK_SINGLECLASS:
                # Chemprop returns 2D array (samples, classes),
                # split into list of 2D arrays (samples, 1), length = n targets
                preds = np.split(preds, preds.shape[1], axis=1)
                # add second column (negative class probability)
                preds = [np.hstack([1 - pred, pred]) for pred in preds]
            else:
                # chemprop returns 2D array (samples, 1), here convert to list and
                # add second column (negative class probability)
                return [np.hstack([1 - preds, preds])]

        return preds

    def loadEstimator(self, params: dict | None = None) -> object:
        """Initialize estimator instance with the given parameters.

        If `params` is `None`, the default parameters will be used.

        Arguments:
            params (dict): algorithm parameters

        Returns:
            object: initialized estimator instance
        """
        self.checkArgs(params)
        new_parameters = self.getParameters(params)
        args = MoleculeModel.getTrainArgs(new_parameters, self.task)

        # set task names
        args.task_names = [prop.name for prop in self.data.targetProperties]

        return self.alg(args)

    def loadEstimatorFromFile(
        self, params: dict | None = None, fallback_load=True
    ) -> object:
        """Load estimator instance from file and apply the given parameters.

        Args:
            params (dict): algorithm parameters
            fallback_load (bool): if `True`, init estimator from alg if path not found

        Returns:
            object: initialized estimator instance
        """
        path = f"{self.outPrefix}.pt"
        # load model state from file
        if os.path.isfile(path):
            if not hasattr(self, "chempropLogger"):
                self.chempropLogger = chemprop.utils.create_logger(
                    name="chemprop_logger",
                    save_dir=self.outDir,
                    quiet=self.quietLogger
                )

            estimator = MoleculeModel.cast(
                chemprop.utils.load_checkpoint(path, logger=self.chempropLogger)
            )
            # load scalers from file and use only the target scaler (first element)
            estimator.scaler = chemprop.utils.load_scalers(path)[0]
            # load parameters from file
            loaded_params = chemprop.utils.load_args(path).as_dict()
            if params is not None:
                loaded_params.update(params)
            self.parameters = self.getParameters(loaded_params)

            # Set train args
            estimator.args = MoleculeModel.getTrainArgs(loaded_params, self.task)
        elif fallback_load:
            self.parameters = self.getParameters(params)
            return self.loadEstimator(params)
        else:
            raise FileNotFoundError(
                f"No estimator found at {path}, loading estimator from file failed."
            )

        return estimator

    def saveEstimator(self) -> str:
        """Save the underlying estimator to file.

        Returns:
            path (str): path to the saved estimator
        """
        chemprop.utils.save_checkpoint(
            f"{self.outPrefix}.pt",
            self.estimator,
            scaler=self.estimator.scaler,
            args=self.estimator.args
        )
        return f"{self.outPrefix}.pt"

    def convertToMoleculeDataset(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        y: pd.DataFrame | np.ndarray | QSPRDataset | None = None
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Convert the given data matrix and target matrix to chemprop Molecule Dataset.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix
            y (pd.DataFrame, np.ndarray, QSPRDataset): target matrix

        Returns:
                data matrix and/or target matrix in np.ndarray format
        """
        if y is not None:
            X, y = self.convertToNumpy(X, y)
            if y.dtype == bool:
                y = y.astype(float)  # BCEWithLogitsLoss expects float
        else:
            X = self.convertToNumpy(X)
            y = [None] * len(X)  # dummy targets

        # find which column contains the SMILES strings
        prev_len = 0
        for calc in self.featureCalculators:
            names = calc.getDescriptorNames()
            if "SMILES" in names:
                smiles_column = names.index("SMILES") + prev_len
                break
            else:
                prev_len += len(names)
        else:
            raise ValueError(
                "No SMILES column found in feature calculators, Chemprop "
                "requires SMILES, make sure to add SMILES calculator to "
                "the feature calculators."
            )

        # features data all but smiles column
        smiles = X[:, smiles_column]
        if X.shape[1] > 1:
            features_data = X[:, np.arange(X.shape[1]) != smiles_column]
            # try to convert to float else raise error
            # Note, in this case features have not been previously converted to float in
            # QSPRpred as SMILES features are not numeric
            try:
                features_data = features_data.astype(np.float32)
            except ValueError:
                raise ValueError(
                    "Features data could not be converted to float, make sure "
                    "that all features are numeric."
                )
        else:
            features_data = None

        # Create MoleculeDataset
        data = chemprop.data.MoleculeDataset(
            [
                chemprop.data.MoleculeDatapoint(
                    smiles=[smile],
                    targets=targets,
                    features=features_data[i] if features_data is not None else None,
                ) for i, (smile, targets) in enumerate(zip(smiles, y))
            ]
        )

        return data

    def cleanFiles(self):
        """Clean up the model files.

        Removes the model directory and all its contents.
        Handles closing the chemprop logger as well.
        """
        handlers = self.chempropLogger.handlers[:]
        for handler in handlers:
            self.chempropLogger.removeHandler(handler)
            handler.close()
        super().cleanFiles()

    def checkArgs(self, args: chemprop.args.TrainArgs | dict):
        """Check if the given arguments are valid.

        Args:
            args (chemprop.args.TrainArgs, dict): arguments to check
        """
        # List of arguments from chemprop that are using in the QSPRpred implementation.
        used_args = [
            "no_cuda", "gpu", "num_workers", "batch_size", "no_cache_mol",
            "empty_cache", "loss_function", "split_sizes", "seed", "pytorch_seed",
            "metric", "bias", "hidden_size", "depth", "mpn_shared", "dropout",
            "activation", "atom_messages", "undirected", "ffn_hidden_size",
            "ffn_num_layers", "explicit_h", "adding_h", "epochs", "warmup_epochs",
            "init_lr", "max_lr", "final_lr", "grad_clip", "class_balance",
            "evidential_regularization", "minimize_score", "num_tasks", "dataset_type",
            "metrics", "task_names"
        ]

        # Create dummy args to check what default argument values are in chemprop
        default_args = chemprop.args.TrainArgs().from_dict(
            args_dict={
                "dataset_type": "regression",
                "data_path": ""
            }
        )
        default_args.process_args()
        default_args = default_args.as_dict()

        # Check if args are valid and warn if changed but not used in QSPRpred
        if isinstance(args, dict) or args is None:
            if isinstance(args, dict):
                for key, value in args.items():
                    if key in default_args:
                        if default_args[key] != value and key not in used_args:
                            print(
                                f"Warning: argument {key} has been set to {value} "
                                f"but is not used in QSPRpred, it will be ignored."
                            )
                    else:
                        print(
                            f"Warning: argument {key} is not a valid argument, it "
                            f"will be ignored."
                        )
            else:
                args = {}

            # add data_path to args as it is required by chemprop
            args["data_path"] = ""

            # set dataset type
            if self.task in [ModelTasks.REGRESSION, ModelTasks.MULTITASK_REGRESSION]:
                args["dataset_type"] = "regression"
            elif self.task in [
                ModelTasks.SINGLECLASS, ModelTasks.MULTITASK_SINGLECLASS
            ]:
                args["dataset_type"] = "classification"
            elif self.task in [ModelTasks.MULTICLASS, ModelTasks.MULTITASK_MULTICLASS]:
                args["dataset_type"] = "multiclass"
            else:
                raise ValueError(f"Task {self.task} not supported.")

            args = chemprop.args.TrainArgs().from_dict(args, skip_unsettable=True)
            args.process_args()

        assert args.split_type in [
            "random", "scaffold_balanced", "random_with_repeated_smiles"
        ], (
            "split_type must be 'random', 'scaffold_balanced' or "
            "random_with_repeated_smiles'."
        )
