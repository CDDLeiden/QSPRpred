"""Wrapper around Chemprop MoleculeModel."""

import chemprop

from ...models.tasks import ModelTasks


class MoleculeModel(chemprop.models.MoleculeModel):
    """Wrapper for chemprop.models.MoleculeModel.

    Attributes:
        args (chemprop.args.TrainArgs): arguments for training the model,
        scaler (chemprop.data.scaler.StandardScaler):
            scaler for scaling the targets
    """
    def __init__(
        self,
        args: chemprop.args.TrainArgs,
        scaler: chemprop.data.scaler.StandardScaler | None = None
    ):
        """Initialize a MoleculeModel instance.

        Args:
            args (chemprop.args.TrainArgs): arguments for training the model,
            scaler (chemprop.data.scaler.StandardScaler):
                scaler for scaling the targets
        """
        super().__init__(args)
        self.args = args
        self.scaler = scaler

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
        obj.scaler = None
        return obj

    @staticmethod
    def getTrainArgs(args: dict | None, task: ModelTasks) -> chemprop.args.TrainArgs:
        """Get a chemprop.args.TrainArgs instance from a dictionary.

        Args:
            args (dict): dictionary of arguments
            task (ModelTasks): task type

        Returns:
            chemprop.args.TrainArgs: arguments for training the model
        """
        # chemprop TrainArgs requires a dictionary with a "data_path" key
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

        # create TrainArgs instance from dictionary
        train_args = chemprop.args.TrainArgs()
        train_args.from_dict(args, skip_unsettable=True)
        train_args.process_args()
        return train_args
