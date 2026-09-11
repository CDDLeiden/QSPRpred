from enum import Enum
from typing import Literal
from qsprpred.utils.serialization import JSONSerializable


class TargetTasks(Enum):
    """Enum representing the type of task the model
    is supposed to perform for a single target property.
    """

    REGRESSION = "REGRESSION"
    SINGLECLASS = "SINGLECLASS"
    MULTICLASS = "MULTICLASS"

    def isClassification(self):
        """Check if the task is a classification task."""
        return self in [self.SINGLECLASS, self.MULTICLASS]

    def isRegression(self):
        """Check if the task is a regression task."""
        return self in [self.REGRESSION]

    def __str__(self):
        """Return the name of the task."""
        return self.name

    @classmethod
    def getTaskFromTh(cls, th: list[float]) -> "TargetTasks":
        """Get the task type from the threshold.

        Args:
            th (list[float]): threshold for a target task

        Returns:
            TargetTasks: task type for the target property
        """
        if len(th) == 1:
            return TargetTasks.SINGLECLASS
        elif len(th) > 3:
            return TargetTasks.MULTICLASS
        else:
            raise ValueError(
                "Threshold list must contain at least 4 values for multi-class classification."
                "Or only a single value for binary classification to infer the task type."
            )


class ModelTasks(Enum):
    """Enum representing the general type of task the
    model is supposed to perform for all target properties.
    """

    REGRESSION = "REGRESSION"
    SINGLECLASS = "SINGLECLASS"
    MULTICLASS = "MULTICLASS"
    MULTITASK_REGRESSION = "MULTITASK_REGRESSION"
    MULTITASK_SINGLECLASS = "MULTITASK_SINGLECLASS"
    MULTITASK_MULTICLASS = "MULTITASK_MULTICLASS"
    MULTITASK_MIXED = "MULTITASK_MIXED"

    def isClassification(self):
        """Check if the task is a classification task."""
        return self in [
            self.SINGLECLASS,
            self.MULTICLASS,
            self.MULTITASK_SINGLECLASS,
            self.MULTITASK_MULTICLASS,
        ]

    def isRegression(self):
        """Check if the task is a regression task."""
        return self in [self.REGRESSION, self.MULTITASK_REGRESSION]

    def isMixed(self):
        """Check if the task is a mixed task."""
        return self in [self.MULTITASK_MIXED]

    def isMultiTask(self):
        """Check if the task is a multitask task."""
        return self in [
            self.MULTITASK_REGRESSION,
            self.MULTITASK_SINGLECLASS,
            self.MULTITASK_MULTICLASS,
            self.MULTITASK_MIXED,
        ]

    def __str__(self):
        """Return the name of the task."""
        return self.name

    @staticmethod
    def getModelTask(target_properties: list):
        """Return the model type for a given list of target properties."""
        if len(target_properties) == 1:
            return ModelTasks[target_properties[0].task.name]
        elif all(
            target_property.task.isRegression() for target_property in target_properties
        ):
            return ModelTasks.MULTITASK_REGRESSION
        elif all(
            target_property.task.isClassification()
            for target_property in target_properties
        ):
            if all(
                target_property.task == TargetTasks.SINGLECLASS
                for target_property in target_properties
            ):
                return ModelTasks.MULTITASK_SINGLECLASS
            else:
                return ModelTasks.MULTITASK_MULTICLASS
        else:
            return ModelTasks.MULTITASK_MIXED


class TargetSpec(JSONSerializable):
    """Target specifications for a single target property.

    Attributes:
        name (str): name of the target property
        task (Literal[TargetTasks.REGRESSION, TargetTasks.SINGLECLASS,
            TargetTasks.MULTICLASS]): task type for the target property
        th (list[float] | None): threshold for the target property, only used for 
            classification tasks
        nClasses (int): number of classes for the target property, only used for
            classification tasks.
    """

    def __init__(
        self,
        name: str,
        task: Literal[TargetTasks.REGRESSION, TargetTasks.SINGLECLASS,
                      TargetTasks.MULTICLASS],
        th: list[float] | None = None,
        n_classes: int | None = None,
    ):
        """Initialize a TargetSpec object.

        Args:
            name (str): name of the target property
            task (Literal[TargetTasks.REGRESSION, TargetTasks.SINGLECLASS,
              TargetTasks.MULTICLASS]): task type for the target property
            th (list[float] | str): threshold for the target property, only used
                for classification tasks. If target property is already discrete,
                n_classes must be specified, otherwise it is inferred from th.
            n_classes (int): number of classes for the target property. Must be
                specified if th is None and the target property is already discrete.
                If th is provided, n_classes is inferred from it.
        """
        self.name = name
        self.task = task
        if task.isClassification():
            self.setTh(th, n_classes)

    @property
    def th(self):
        """Set the threshold for the target property.

        Returns:
            th ([list[float] | None): threshold for the target property
        """
        if not self.task.isClassification():
            raise AttributeError("Threshold is only available for classification tasks")
        return self._th
    
    @property
    def nClasses(self):
        """Get the number of classes for the target property.

        Returns:
            nClasses (int): number of classes
        """
        assert self.task.isClassification(), "Number of classes is only available for classification tasks"
        return self._nClasses

    def setTh(self, th: list[float] | None, n_classes: int | None = None):
        """Set the threshold for the target property and the number of classes if th is
        precomputed.

        Args:
            th (list[float] | None): threshold for the target property
            n_classes (int | None): number of classes for the target property
        """
        assert (
            self.task.isClassification()
        ), "Threshold can only be set for classification tasks"
        self._th = th
        if self.th is None:
            assert n_classes is not None, (
                "If target property is a classification task, "
                "either a threshold or the number of classes must be specified."
                "Make sure to set nClasses first if setting th to None."
            )
            self._nClasses = n_classes
        else:
            assert isinstance(self.th, list), "Threshold must be a list of floats."
            assert len(self.th) > 0, "Threshold list must contain at least one value."
            assert n_classes is None, (
                "If th is provided, n_classes must be None. "
                "Number of classes is inferred from the threshold."
            )
            if len(self.th) > 1:
                assert self.task == TargetTasks.MULTICLASS, (
                    f"If multiple thresholds are provided, "
                    f"task must be {TargetTasks.MULTICLASS}, "
                    f"but got {self.task}."
                )
                assert len(self.th) > 3, (
                    "For multi-class classification, set at least 4 thresholds. "
                    "These define the lower and upper bounds of the bins, e.g. "
                    "[1, 2, 3, 4] will create bins (1,2], (2,3], (3,4]. "
                    "For binary classification, set a single threshold."
                )
            else:
                assert self.task == TargetTasks.SINGLECLASS, (
                    f"If a single threshold is provided, "
                    f"task must be {TargetTasks.SINGLECLASS}, "
                    f"but got {self.task}."
                )
            self._nClasses = len(self.th) - 1 if len(self.th) > 1 else 2

    @th.deleter
    def th(self):
        """Delete the threshold for the target property and the number of classes."""
        del self._th
        del self._nClasses

    def __repr__(self):
        """Representation of the TargetSpec object."""
        if self.task.isClassification():
            return f"TargetSpec(name={self.name}, task={self.task}, th={self.th}, n_classes={self.nClasses})"
        else:
            return f"TargetSpec(name={self.name}, task={self.task})"

    def __str__(self):
        """Return string identifier of the TargetSpec object."""
        return self.name

    @classmethod
    def fromDict(cls, d: dict[str, str | list[float] | int]):
        """Create a TargetSpec object from a dictionary.

        task can be specified as a string or as a TargetTasks object.

        Args:
            d (dict): dictionary containing the target property information

        Example:
            >>> TargetSpec.fromDict({"name": "property_name", "task": "regression"})
            TargetSpec(name=property_name, task=REGRESSION)

        Returns:
            TargetSpec: TargetSpec object
        """
        if isinstance(d["task"], str):
            d["task"] = TargetTasks[d["task"].upper()]
        return TargetSpec(**d)

    @classmethod
    def fromList(cls, _list: list[dict]):
        """Create a list of TargetSpec objects from a list of dictionaries.

        Args:
            _list (list): list of dictionaries containing the target property
                information

        Returns:
            list[TargetSpec]: list of TargetSpec objects
        """
        return [cls.fromDict(d) for d in _list]

    @staticmethod
    def toList(_list: list, task_as_str: bool = False):
        """Convert a list of TargetSpec objects to a list of dictionaries.

        Args:
            _list (list): list of TargetSpec objects
            task_as_str (bool): whether to convert the task to a string

        Returns:
            list[dict]: list of dictionaries containing the target property information
        """
        target_props = []
        for target_prop in _list:
            target_props.append(
                {
                    "name": target_prop.name,
                    "task": target_prop.task.name if task_as_str else target_prop.task,
                }
            )
            if target_prop.task.isClassification():
                target_props[-1].update(
                    {
                        "th": target_prop.th,
                        "n_classes": target_prop.nClasses
                    }
                )
        return target_props

    @staticmethod
    def selectFromList(_list: list, names: list):
        """Select a subset of TargetSpec objects from a list of TargetSpec
        objects.

        Args:
            _list (list): list of TargetSpec objects
            names (list): list of names of the target properties to be selected
            original_names (bool): whether to use the original names of the target
                properties

        Returns:
            list[TargetSpec]: list of TargetSpec objects
        """
        return [t for t in _list if t.name in names]

    @staticmethod
    def getNames(_list: list):
        """Get the names of the target properties from a list of TargetSpec objects.

        Args:
            _list (list): list of TargetSpec objects

        Returns:
            list[str]: list of names of the target properties
        """
        return [t.name for t in _list]
