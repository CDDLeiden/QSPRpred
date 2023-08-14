"""QSPRpred implementation of Chemprop model."""

import os
from copy import deepcopy
from typing import Any
import shutil

import chemprop
import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm, trange

from ...data.data import QSPRDataset
from ...models.interfaces import QSPRModel
from ...models.tasks import ModelTasks
from ...models.early_stopping import EarlyStoppingMode, early_stopping


class MoleculeModel(chemprop.models.MoleculeModel):
    """Wrapper for chemprop.models.MoleculeModel."""
    def __init__(
        self,
        args: chemprop.args.TrainArgs,
        scaler: chemprop.data.scaler.StandardScaler | None = None,
        features_scaler: chemprop.data.scaler.StandardScaler | None = None,
        atom_descriptor_scaler: chemprop.data.scaler.StandardScaler | None = None,
        bond_descriptor_scaler: chemprop.data.scaler.StandardScaler | None = None,
        atom_bond_scaler: chemprop.data.scaler.StandardScaler | None = None,
    ):
        """Initialize a MoleculeModel instance.

        self.parameters:
            args (chemprop.args.TrainArgs): arguments for training the model,
            scaler (chemprop.data.scaler.StandardScaler):
                scaler for scaling the targets
            features_scaler (chemprop.data.scaler.StandardScaler):
                scaler for scaling the features
            atom_descriptor_scaler (chemprop.data.scaler.StandardScaler):
                scaler for scaling the atom descriptors
            bond_descriptor_scaler (chemprop.data.scaler.StandardScaler):
                scaler for scaling the bond descriptors
            atom_bond_scaler (chemprop.data.scaler.StandardScaler):
                scaler for scaling the atom/bond targets,
                unused in QSPRpred
        """
        super().__init__(args)
        self.args = args
        self.scaler = scaler
        self.features_scaler = features_scaler
        self.atom_descriptor_scaler = atom_descriptor_scaler
        self.bond_descriptor_scaler = bond_descriptor_scaler
        self.atom_bond_scaler = atom_bond_scaler

    def setScalers(
        self,
        scaler: chemprop.data.scaler.StandardScaler | None = None,
        features_scaler: chemprop.data.scaler.StandardScaler | None = None,
        atom_descriptor_scaler: chemprop.data.scaler.StandardScaler | None = None,
        bond_descriptor_scaler: chemprop.data.scaler.StandardScaler | None = None,
        atom_bond_scaler: chemprop.data.scaler.StandardScaler | None = None,
    ):
        """Set the scalers of the model.

        self.parameters:
            scaler (chemprop.data.scaler.StandardScaler):
                scaler for scaling the targets
            features_scaler (chemprop.data.scaler.StandardScaler):
                scaler for scaling the features
            atom_descriptor_scaler (chemprop.data.scaler.StandardScaler):
                scaler for scaling the atom descriptors
            bond_descriptor_scaler (chemprop.data.scaler.StandardScaler):
                scaler for scaling the bond descriptors
            atom_bond_scaler (chemprop.data.scaler.StandardScaler):
                scaler for scaling the atom/bond targets,
                unused in QSPRpred
        """
        self.scaler = scaler
        self.features_scaler = features_scaler
        self.atom_descriptor_scaler = atom_descriptor_scaler
        self.bond_descriptor_scaler = bond_descriptor_scaler
        self.atom_bond_scaler = atom_bond_scaler

    def getScalers(self):
        return [
            self.scaler, self.features_scaler, self.atom_descriptor_scaler,
            self.bond_descriptor_scaler, self.atom_bond_scaler
        ]

    @classmethod
    def cast(cls, obj: chemprop.models.MoleculeModel) -> "MoleculeModel":
        """Cast a chemprop.models.MoleculeModel instance to a MoleculeModel instance.

        Args:
            obj (chemprop.models.MoleculeModel): instance to cast

        Returns:
            MoleculeModel: casted MoleculeModel instance
        """
        assert isinstance(
            obj, chemprop.models.MoleculeModel
        ), "obj is not a chemprop.models.MoleculeModel instance."
        obj.__class__ = cls
        obj.args = None
        obj.setScalers()
        return obj

    @staticmethod
    def getTrainArgs(args: dict | None, task: ModelTasks) -> chemprop.args.TrainArgs:
        """Get a chemprop.args.TrainArgs instance from a dictionary.

        Args:
            args (dict): dictionary of arguments

        Returns:
            chemprop.args.TrainArgs: arguments for training the model
        """
        if args is None:
            args = {"data_path": ""}

        # set dataset type
        if task in [ModelTasks.REGRESSION, ModelTasks.MULTITASK_REGRESSION]:
            args["dataset_type"] = "regression"
        elif task in [ModelTasks.SINGLECLASS, ModelTasks.MULTITASK_SINGLECLASS]:
            args["dataset_type"] = "classification"
        elif task in [ModelTasks.MULTICLASS, ModelTasks.MULTITASK_MULTICLASS]:
            args["dataset_type"] = "multiclass"
        else:
            raise ValueError(f"Task {task} not supported.")

        train_args = chemprop.args.TrainArgs()
        train_args.from_dict(args, skip_unsettable=True)
        train_args.process_args()
        return train_args


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
        """Initialize a QSPR model instance.

        If the model is loaded from file, the data set is not required.
        Note that the data set is required for fitting and optimization.

        self.parameters:
            base_dir (str):
                base directory of the model,
                the model files are stored in a subdirectory `{baseDir}/{outDir}/`
            data (QSPRDataset): data set used to train the model
            name (str): name of the model
            parameters (dict): dictionary of algorithm specific parameters
            autoload (bool):
                if `True`, the estimator is loaded from the serialized file
                if it exists, otherwise a new instance of alg is created
            quiet_logger (bool):
                if `True`, the logger is set to quiet mode (no debug messages)
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
        if self.chempropLogger is not None:
            debug, info = self.chempropLogger.debug, self.chempropLogger.info
        else:
            debug = info = print

        estimator = self.estimator if estimator is None else estimator

        data = self.convertToMoleculeDataset(estimator, X, y)
        args = estimator.args

        # Set pytorch seed for random initial weights
        torch.manual_seed(estimator.args.pytorch_seed)

        # Split data
        debug(f"Splitting data with seed {args.seed}")

        # Set pytorch seed for random initial weights
        torch.manual_seed(args.pytorch_seed)

        # set task names
        args.task_names = [prop.name for prop in self.data.targetProperties]

        if self.earlyStopping:
            # Split data
            debug(f"Splitting data with seed {args.seed}")
            train_data, val_data, _ = chemprop.data.utils.split_data(
                data=data,
                split_type=args.split_type,
                sizes=args.split_sizes if args.split_sizes[2] == 0 else [0.9, 0.1, 0],
                key_molecule_index=args.split_key_molecule,
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

        # Fit feature scaler on train data and scale data
        if args.features_scaling:
            features_scaler = train_data.normalize_features(replace_nan_token=0)
            if self.earlyStopping:
                val_data.normalize_features(features_scaler)
        else:
            features_scaler = None

        # Fit atom descriptor scaler on train data and scale data
        if args.atom_descriptor_scaling and args.atom_descriptors is not None:
            atom_descriptor_scaler = train_data.normalize_features(
                replace_nan_token=0, scale_atom_descriptors=True
            )
            if self.earlyStopping:
                val_data.normalize_features(
                    atom_descriptor_scaler, scale_atom_descriptors=True
                )
        else:
            atom_descriptor_scaler = None

        # Fit bond descriptor scaler on train data and scale data
        if args.bond_descriptor_scaling and args.bond_descriptors is not None:
            bond_descriptor_scaler = train_data.normalize_features(
                replace_nan_token=0, scale_bond_descriptors=True
            )
            if self.earlyStopping:
                val_data.normalize_features(
                    bond_descriptor_scaler, scale_bond_descriptors=True
                )
        else:
            bond_descriptor_scaler = None

        # Get length of training data
        args.train_data_size = len(train_data)

        # log data size
        debug(f"Total size = {len(data):,}")
        if self.earlyStopping:
            debug(f"train size = {len(train_data):,} | val size = {len(val_data):,}")

        # Initialize scaler and standard scale training targets (regression only)
        if args.dataset_type == "regression":
            debug("Fitting scaler")
            scaler = train_data.normalize_targets()
        else:
            scaler = None

        # attach scalers to estimator
        estimator.setScalers(
            scaler, features_scaler, atom_descriptor_scaler, bond_descriptor_scaler,
            None
        )

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
        save_dir = os.path.join(self.outDir, "temp")
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
                    scaler=scaler,
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
        estimator: Any = None
    ) -> np.ndarray:
        """Make predictions for the given data matrix or `QSPRDataset`.

        Note. convertToNumpy can be called here, to convert the input data to
        np.ndarray format.

        Note. if no estimator is given, the estimator instance of the model
              is used.

        self.parameters:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to predict
            estimator (Any): estimator instance to use for fitting

        Returns:
            np.ndarray:
                2D array containing the predictions, where each row corresponds
                to a sample in the data and each column to a target property
        """
        if self.task.isClassification():
            preds = self.predictProba(X, estimator)
            preds = [
                np.argmax(preds[i], axis=1, keepdims=True) for i in range(len(preds))
            ]
            # change preds from list of 2D arrays to 2D array
            preds = np.concatenate(preds, axis=1)
            print(preds.shape)
            return preds
        return self.predictProba(X, estimator)

    def predictProba(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any = None
    ) -> list[np.ndarray]:
        """Make predictions for the given data matrix or `QSPRDataset`,
        but use probabilities for classification models. Does not work with
        regression models.

        Note. convertToNumpy can be called here, to convert the input data to
        np.ndarray format.

        Note. if no estimator is given, the estimator instance of the model
              is used.

        self.parameters:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to make predict
            estimator (Any): estimator instance to use for fitting

        Returns:
            list[np.ndarray]:
                a list of 2D arrays containing the probabilities for each class,
                where each array corresponds to a target property, each row
                to a sample in the data and each column to a class
        """
        estimator = self.estimator if estimator is None else estimator
        X = self.convertToMoleculeDataset(estimator, X)
        args = estimator.args

        if args.features_scaling:
            X.normalize_features(estimator.features_scaler)
        if args.atom_descriptor_scaling and args.atom_descriptors is not None:
            X.normalize_features(
                estimator.atom_descriptor_scaler, scale_atom_descriptors=True
            )
        if args.bond_descriptor_scaling and args.bond_descriptors_size > 0:
            X.normalize_features(
                estimator.bond_descriptor_scaler,
                scale_bond_descriptors=True,
            )

        X_loader = chemprop.data.MoleculeDataLoader(
            dataset=X, batch_size=args.batch_size
        )

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
                print(preds.shape)
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
        new_parameters = self.getParameters(params)
        self.checkArgs(new_parameters)
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
            # load scalers from file
            estimator.setScalers(*chemprop.utils.load_scalers(path))
            # load parameters from file
            loaded_params = chemprop.utils.load_args(path).as_dict()
            if params is not None:
                loaded_params.update(params)
            self.checkArgs(loaded_params)
            self.parameters = self.getParameters(loaded_params)

            # Set train args
            estimator.args = MoleculeModel.getTrainArgs(loaded_params, self.task)
        elif fallback_load:
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
            f"{self.outPrefix}.pt", self.estimator, *self.estimator.getScalers(),
            self.estimator.args
        )
        return f"{self.outPrefix}.pt"

    def convertToMoleculeDataset(
        self,
        estimator: MoleculeModel,
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
                )
                for i, (smile,
                        targets) in tqdm(enumerate(zip(smiles, y)), total=len(smiles))
            ]
        )

        return data

    def cleanFiles(self):
        handlers = self.chempropLogger.handlers[:]
        for handler in handlers:
            self.chempropLogger.removeHandler(handler)
            handler.close()
        return super().cleanFiles()

    @staticmethod
    def checkArgs(args: chemprop.args.TrainArgs | dict):
        """Check if the given arguments are valid.

        Args:
            args (chemprop.args.TrainArgs, dict): arguments to check
        """
        unused_args = [
            "smiles_columns",
            "number_of_molecules",
            "checkpoint_dir",
            "checkpoint_path",
            "checkpoint_paths",
            "gpu",
            "phase_features_path",
            "max_data_size",
            "num_workers",  # common args
            "data_path",
            "target_columns",
            "ignore_columns",
            "dataset_type",
            "multiclass_num_classes",
            "separate_val_path",
            "separate_test_path",
            "spectra_phase_mask_path",
            "data_weights_path",
            "target_weights",
            "split_type",
            "split_key_molecule",
            "num_folds",
            "folds_file",
            "val_fold_index",
            "test_fold_index",
            "crossval_index_dir",
            "crossval_index_file",
            "extra_metrics",
            "save_dir",
            "save_smiles_split",
            "test",
            "quiet",
            "log_frequency",
            "show_individual_scores",
            "cache_cutoff",
            "save_preds",
            "resume_experiment"  # general train args
            "separate_val_features_path",
            "separate_test_features_path",
            "separate_val_phase_features_path",
            "separate_test_phase_features_path",
            "separate_val_atom_descriptors_path",
            "separate_test_atom_descriptors_path",
            "separate_val_bond_features_path",
            "separate_test_bond_features_path",
            "config_path",
            "ensemble_size",
            "aggregation",
            "aggregation_norm",
            "reaction",
            "reaction_mode",
            "reaction_solvent"  # model train args
            "frzn_ffn_layers",
            "freeze_first_only"  # training train args
        ]
        if isinstance(args, dict) or args is None:
            if isinstance(args, dict):
                # check if any key in args is in unused_common_args
                set_unused_args = [key in unused_args for key in args.keys()]
                if any(set_unused_args):
                    print(
                        "Warning: unused common arguments sets: "
                        f"{np.array(list(args.keys()))[set_unused_args]}"
                    )
            else:
                args = {}
            # add data_path to args
            args["data_path"] = ""

            if "dataset_type" not in args.keys():
                args["dataset_type"] = "regression"

            args = chemprop.args.TrainArgs().from_dict(args, skip_unsettable=True)
            args.process_args()

        assert args.split_key_molecule == 0, (
            "split_key_molecule must be 0, as QSPRpred does not support data with "
            "multiple molecules, i.e. reactions."
        )

        assert args.split_type in [
            "random", "scaffold_balanced", "random_with_repeated_smiles"
        ], (
            "split_type must be 'random', 'scaffold_balanced' or "
            "random_with_repeated_smiles'."
        )
