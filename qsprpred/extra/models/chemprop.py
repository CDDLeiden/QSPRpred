"""QSPRpred implementation of Chemprop model."""

import os
from logging import Logger
from typing import Any, Dict, List, Optional

import chemprop
import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from ...data.data import QSPRDataset
from ...models.interfaces import QSPRModel


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
        obj.setScalers()
        return obj

    @staticmethod
    def getTrainArgs(args: dict) -> chemprop.args.TrainArgs:
        """Get a chemprop.args.TrainArgs instance from a dictionary.

        Args:
            args (dict): dictionary of arguments

        Returns:
            chemprop.args.TrainArgs: arguments for training the model
        """
        train_args = chemprop.args.TrainArgs()
        train_args.from_dict(args, skip_unsettable=True)
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
        autoload=True
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
        """
        alg = MoleculeModel  # wrapper for chemprop.models.MoleculeModel
        super().__init__(base_dir, alg, data, name, parameters, autoload)
        self.chempropLogger = chemprop.utils.create_logger(
            name=chemprop.constants.TRAIN_LOGGER_NAME,
            save_dir=self.baseDir,
            quiet=True
        )

    def supportsEarlyStopping(self) -> bool:
        """Return if the model supports early stopping.

        Returns:
            bool: True if the model supports early stopping
        """
        return True

    def fit(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        y: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any = None,
        early_stopping: bool | None = None
    ) -> Any | tuple[Any, int] | None:
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
            int]: in case of early stopping, the number of iterations
                after which the model stopped training
        """
        raise NotImplementedError("Not implemented yet.")

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
        raise NotImplementedError("Not implemented yet.")

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
        raise NotImplementedError("Not implemented yet.")

    def loadEstimator(self, params: dict | None = None) -> object:
        """Initialize estimator instance with the given parameters.

        If `params` is `None`, the default parameters will be used.

        Arguments:
            params (dict): algorithm parameters

        Returns:
            object: initialized estimator instance
        """
        new_parameters = self.getParameters(params)
        return self.alg(MoleculeModel.getTrainArgs(new_parameters))

    def loadEstimatorFromFile(self, params: dict | None = None) -> object:
        """Load estimator instance from file and apply the given parameters.

        self.parameters:
            params (dict): algorithm parameters

        Returns:
            object: initialized estimator instance
        """
        path = f"{self.outPrefix}.pt"
        # load model state from file
        if os.path.isfile(path):
            estimator = MoleculeModel.cast(
                chemprop.utils.load_checkpoint(path, logger=self.chempropLogger)
            )
            # load scalers from file
            estimator.setScalers(chemprop.utils.load_scalers(path))
            # load parameters from file
            loaded_params = chemprop.utils.load_args(path).as_dict()
            if params is not None:
                loaded_params.update(params)
            self.parameters = self.getParameters(loaded_params)

            # Set train args
            estimator.args = MoleculeModel.getTrainArgs(loaded_params)
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

    def run_training(
        args: chemprop.args.TrainArgs,
        data: chemprop.data.MoleculeDataset,
        logger: Optional[Logger] = None
    ) -> Dict[str, List[float]]:
        """
        Loads data, trains a Chemprop model, and returns test scores for the model checkpoint with the highest validation score.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                    loading data and training the Chemprop model.
        :param data: A :class:`~chemprop.data.MoleculeDataset` containing the data.
        :param logger: A logger to record output.
        :return: A dictionary mapping each metric in :code:`args.metrics` to a list of values for each task.

        """
        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = print

        # Set pytorch seed for random initial weights
        torch.manual_seed(args.pytorch_seed)

        # Split data
        debug(f"Splitting data with seed {args.seed}")
        train_data, val_data, _ = chemprop.data.utils.split_data(
            data=data,
            split_type=args.split_type,
            sizes=args.split_sizes,
            key_molecule_index=args.split_key_molecule,
            seed=args.seed,
            num_folds=args.num_folds,
            args=args,
            logger=logger
        )

        # Get number of molecules per class in training data
        if args.dataset_type == "classification":
            class_sizes = chemprop.data.utils.get_class_sizes(data)
            debug("Class sizes")
            for i, task_class_sizes in enumerate(class_sizes):
                debug(
                    f"{args.task_names[i]} "
                    f"{', '.join(f'{cls}: {size * 100:.2f}%' for cls, size in enumerate(task_class_sizes))}"
                )
            train_class_sizes = chemprop.data.utils.get_class_sizes(
                train_data, proportion=False
            )
            args.train_class_sizes = train_class_sizes

        # Fit feature scaler on train data and scale data
        if args.features_scaling:
            features_scaler = train_data.normalize_features(replace_nan_token=0)
            val_data.normalize_features(features_scaler)
        else:
            features_scaler = None

        # Fit atom descriptor scaler on train data and scale data
        if args.atom_descriptor_scaling and args.atom_descriptors is not None:
            atom_descriptor_scaler = train_data.normalize_features(
                replace_nan_token=0, scale_atom_descriptors=True
            )
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
            val_data.normalize_features(
                bond_descriptor_scaler, scale_bond_descriptors=True
            )
        else:
            bond_descriptor_scaler = None

        # Get length of training data
        args.train_data_size = len(train_data)

        debug(
            f"Total size = {len(data):,} | "
            f"train size = {len(train_data):,} | val size = {len(val_data):,}"
        )

        if len(val_data) == 0:
            raise ValueError(
                "The validation data split is empty. During normal chemprop training (non-sklearn functions), \
                a validation set is required to conduct early stopping according to the selected evaluation metric. This \
                may have occurred because validation data provided with `--separate_val_path` was empty or contained only invalid molecules."
            )

        # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
        if args.dataset_type == "regression":
            debug("Fitting scaler")
            if args.is_atom_bond_targets:
                scaler = None
                atom_bond_scaler = train_data.normalize_atom_bond_targets()
            else:
                scaler = train_data.normalize_targets()
                atom_bond_scaler = None
        else:
            scaler = None
            atom_bond_scaler = None

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
        val_data_loader = chemprop.data.MoleculeDataLoader(
            dataset=val_data, batch_size=args.batch_size, num_workers=num_workers
        )

        if args.class_balance:
            debug(
                f"With class_balance, effective train size = {train_data_loader.iter_size:,}"
            )

        # Train ensemble of models
        for model_idx in range(args.ensemble_size):
            # Tensorboard writer
            save_dir = os.path.join(args.save_dir, f"model_{model_idx}")
            os.makedirs(save_dir)
            try:
                writer = SummaryWriter(log_dir=save_dir)
            except:  # noqa: E722
                writer = SummaryWriter(logdir=save_dir)

            # Load/build model
            if args.checkpoint_paths is not None:
                debug(
                    f"Loading model {model_idx} from {args.checkpoint_paths[model_idx]}"
                )
                model = chemprop.utils.load_checkpoint(
                    args.checkpoint_paths[model_idx], logger=logger
                )
            else:
                debug(f"Building model {model_idx}")
                model = chemprop.models.MoleculeModel(args)

            # Optionally, overwrite weights:
            if args.checkpoint_frzn is not None:
                debug(f"Loading and freezing parameters from {args.checkpoint_frzn}.")
                model = chemprop.utils.load_frzn_model(
                    model=model,
                    path=args.checkpoint_frzn,
                    current_args=args,
                    logger=logger
                )

            debug(model)

            if args.checkpoint_frzn is not None:
                debug(
                    f"Number of unfrozen parameters = {chemprop.nn_utils.param_count(model):,}"
                )
                debug(
                    f"Total number of parameters = {chemprop.nn_utils.param_count_all(model):,}"
                )
            else:
                debug(
                    f"Number of parameters = {chemprop.nn_utils.param_count_all(model):,}"
                )

            if args.cuda:
                debug("Moving model to cuda")
            model = model.to(args.device)

            # Ensure that model is saved in correct location for evaluation if 0 epochs
            chemprop.utils.save_checkpoint(
                os.path.join(save_dir, chemprop.constants.MODEL_FILE_NAME), model,
                scaler, features_scaler, atom_descriptor_scaler, bond_descriptor_scaler,
                atom_bond_scaler, args
            )

            # Optimizers
            optimizer = chemprop.utils.build_optimizer(model, args)

            # Learning rate schedulers
            scheduler = chemprop.utils.build_lr_scheduler(optimizer, args)

            # Run training
            best_score = float("inf") if args.minimize_score else -float("inf")
            best_epoch, n_iter = 0, 0
            for epoch in trange(args.epochs):
                debug(f"Epoch {epoch}")
                n_iter = chemprop.train.train(
                    model=model,
                    data_loader=train_data_loader,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    n_iter=n_iter,
                    atom_bond_scaler=atom_bond_scaler,
                    logger=logger,
                    writer=writer
                )
                if isinstance(scheduler, ExponentialLR):
                    scheduler.step()
                val_scores = chemprop.train.evaluate(
                    model=model,
                    data_loader=val_data_loader,
                    num_tasks=args.num_tasks,
                    metrics=args.metrics,
                    dataset_type=args.dataset_type,
                    scaler=scaler,
                    atom_bond_scaler=atom_bond_scaler,
                    logger=logger
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
                    chemprop.utils.save_checkpoint(
                        os.path.join(save_dir, chemprop.constants.MODEL_FILE_NAME),
                        model, scaler, features_scaler, atom_descriptor_scaler,
                        bond_descriptor_scaler, atom_bond_scaler, args
                    )

            # Evaluate on test set using model with best validation score
            info(
                f"Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}"
            )
            return chemprop.utils.load_checkpoint(
                os.path.join(save_dir, chemprop.constants.MODEL_FILE_NAME),
                device=args.device,
                logger=logger
            )
