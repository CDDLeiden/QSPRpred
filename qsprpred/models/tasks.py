"""Enums representing what type of task the model is supposed to perform for a single target property and for all target properties."""
from enum import Enum


class TargetTasks(Enum):
    """Enum representing the type of task the model is supposed to perform for a single target property."""

    REGRESSION = 'REGRESSION'
    SINGLECLASS = 'SINGLECLASS'
    MULTICLASS = 'MULTICLASS'

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
    """Enum representing the general type of task the model is supposed to perform for all target properties."""

    REGRESSION = 'REGRESSION'
    SINGLECLASS = 'SINGLECLASS'
    MULTICLASS = 'MULTICLASS'
    MULTITASK_REGRESSION = 'MULTITASK_REGRESSION'
    MULTITASK_SINGLECLASS = 'MULTITASK_SINGLECLASS'
    MULTITASK_MULTICLASS = 'MULTITASK_MULTICLASS'
    MULTITASK_MIXED = 'MULTITASK_MIXED'

    def isClassification(self):
        """Check if the task is a classification task."""
        return self in [self.SINGLECLASS, self.MULTICLASS, self.MULTITASK_SINGLECLASS, self.MULTITASK_MULTICLASS]

    def isRegression(self):
        """Check if the task is a regression task."""
        return self in [self.REGRESSION, self.MULTITASK_REGRESSION]

    def isMixed(self):
        """Check if the task is a mixed task."""
        return self in [self.MULTITASK_MIXED]

    def isMultiTask(self):
        """Check if the task is a multitask task."""
        return self in [self.MULTITASK_REGRESSION, self.MULTITASK_SINGLECLASS,
                        self.MULTITASK_MULTICLASS, self.MULTITASK_MIXED]

    def __str__(self):
        """Return the name of the task."""
        return self.name

    @staticmethod
    def getModelTask(target_properties: list):
        """Return the model type for a given list of target properties."""
        if len(target_properties) == 1:
            return ModelTasks[target_properties[0].task.name]
        else:
            if all([target_property.task.isRegression() for target_property in target_properties]):
                return ModelTasks.MULTITASK_REGRESSION
            elif all([target_property.task.isClassification() for target_property in target_properties]):
                if all([target_property.task == TargetTasks.SINGLECLASS for target_property in target_properties]):
                    return ModelTasks.MULTITASK_SINGLECLASS
                else:
                    return ModelTasks.MULTITASK_MULTICLASS
            else:
                return ModelTasks.MULTITASK_MIXED
