from enum import Enum
from typing import Literal, Optional, Callable

from qsprpred.utils.serialization import (
    JSONSerializable,
    function_as_string,
    function_from_string,
)


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


class TargetProperty(JSONSerializable):
    """Target property for QSPRmodelling class.

    Attributes:
        name (str): name of the target property
        task (Literal[TargetTasks.REGRESSION,
              TargetTasks.SINGLECLASS,
              TargetTasks.MULTICLASS]): task type for the target property
        th (int): threshold for the target property, only used for classification tasks
        nClasses (int): number of classes for the target property, only used for
            classification tasks
        transformer (Callable): function to transform the target property
        imputer (Callable): function to impute the target property
    """

    _notJSON = ["transformer", *JSONSerializable._notJSON]

    def __init__(
        self,
        name: str,
        task: Literal[
            TargetTasks.REGRESSION, TargetTasks.SINGLECLASS, TargetTasks.MULTICLASS
        ],
        th: Optional[list[float] | str] = None,
        n_classes: Optional[int] = None,
        transformer: Optional[Callable] = None,
        imputer: Optional[Callable] = None,
    ):
        """Initialize a TargetProperty object.

        Args:
            name (str): name of the target property
            task (Literal[TargetTasks.REGRESSION,
              TargetTasks.SINGLECLASS,
              TargetTasks.MULTICLASS]): task type for the target property
            th (list[float] | str): threshold for the target property, only used
                for classification tasks. If th is precomputed, set it to "precomputed".
                If th is precomputed, n_classes must be specified.
            n_classes (int): number of classes for the target property. Must be
                specified if th is precomputed, otherwise it is inferred from th.
            transformer (Callable): function to transform the target property
            imputer (Callable): function to impute the target property
        """
        self.name = name
        self.task = task
        if task.isClassification():
            assert (
                th is not None
            ), (f"Threshold not specified for classification task `{name}`. "
                "If the task is already precomputed, set `th` to `precomputed`, and "
                "define the correct number of classes with `n_classes."
                )
            self.th = th
            if isinstance(th, str) and th == "precomputed":
                self.nClasses = n_classes
        self.transformer = transformer
        self.imputer = imputer

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["transformer"] = function_as_string(self.transformer) if self.transformer else None
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        if state["transformer"] is not None:
            self.transformer = function_from_string(self.transformer)

    @property
    def th(self):
        """Set the threshold for the target property.

        Returns:
            th ([list[int] | str]): threshold for the target property
        """
        return self._th

    @th.setter
    def th(self, th: list[float] | str):
        """Set the threshold for the target property and the number of classes if th is
        not precomputed.

        Args:
            th (list[float] | str): threshold for the target property
        """
        assert (
            self.task.isClassification()
        ), "Threshold can only be set for classification tasks"
        self._th = th
        if isinstance(th, str):
            assert th == "precomputed", f"Invalid threshold {th}"
        else:
            self._nClasses = len(self.th) - 1 if len(self.th) > 1 else 2

    @th.deleter
    def th(self):
        """Delete the threshold for the target property and the number of classes."""
        del self._th
        del self._nClasses

    @property
    def nClasses(self):
        """Get the number of classes for the target property.

        Returns:
            nClasses (int): number of classes
        """
        return self._nClasses

    @nClasses.setter
    def nClasses(self, nClasses: int):
        """Set the number of classes for the target property if th is precomputed.

        Args:
            nClasses (int): number of classes
        """
        assert (
            self.th == "precomputed"
        ), "Number of classes can only be set if threshold is precomputed"
        self._nClasses = nClasses

    def __repr__(self):
        """Representation of the TargetProperty object."""
        if self.task.isClassification() and self.th is not None:
            return f"TargetProperty(name={self.name}, task={self.task}, th={self.th})"
        else:
            return f"TargetProperty(name={self.name}, task={self.task})"

    def __str__(self):
        """Return string identifier of the TargetProperty object."""
        return self.name

    @classmethod
    def fromDict(cls, d: dict):
        """Create a TargetProperty object from a dictionary.

        task can be specified as a string or as a TargetTasks object.

        Args:
            d (dict): dictionary containing the target property information

        Example:
            >>> TargetProperty.fromDict({"name": "property_name", "task": "regression"})
            TargetProperty(name=property_name, task=REGRESSION)

        Returns:
            TargetProperty: TargetProperty object
        """
        if isinstance(d["task"], str):
            return TargetProperty(
                **{
                    k: TargetTasks[v.upper()] if k == "task" else v
                    for k, v in d.items()
                }
            )
        else:
            return TargetProperty(**d)

    @classmethod
    def fromList(cls, _list: list[dict]):
        """Create a list of TargetProperty objects from a list of dictionaries.

        Args:
            _list (list): list of dictionaries containing the target property
                information

        Returns:
            list[TargetProperty]: list of TargetProperty objects
        """
        return [cls.fromDict(d) for d in _list]

    @staticmethod
    def toList(_list: list, task_as_str: bool = False, drop_transformer: bool = True):
        """Convert a list of TargetProperty objects to a list of dictionaries.

        Args:
            _list (list): list of TargetProperty objects
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
                    {"th": target_prop.th, "n_classes": target_prop.nClasses}
                )
            if not drop_transformer:
                target_props[-1].update({"transformer": target_prop.transformer})
        return target_props

    @staticmethod
    def selectFromList(_list: list, names: list):
        """Select a subset of TargetProperty objects from a list of TargetProperty
        objects.

        Args:
            _list (list): list of TargetProperty objects
            names (list): list of names of the target properties to be selected
            original_names (bool): whether to use the original names of the target
                properties

        Returns:
            list[TargetProperty]: list of TargetProperty objects
        """
        return [t for t in _list if t.name in names]

    @staticmethod
    def getNames(_list: list):
        """Get the names of the target properties from a list of TargetProperty objects.

        Args:
            _list (list): list of TargetProperty objects

        Returns:
            list[str]: list of names of the target properties
        """
        return [t.name for t in _list]
