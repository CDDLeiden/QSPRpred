"""
tasks

Created by: Martin Sicho
On: 28.11.22, 15:23
"""
from enum import Enum


class ModelTasks(Enum):
    """Enum representing the type of task the model is supposed to perform for a single target property."""

    REGRESSION = 'REGRESSION'
    SINGLECLASS = 'SINGLECLASS'
    MULTICLASS = 'MULTICLASS'

    def isClassification(self):
        """Check if the task is a classification task.""""
        return self in [self.SINGLECLASS, self.MULTICLASS]

    def __str__(self):
        """Return the name of the task."""
        return self.name


class ModelTypes(Enum):
    """Enum representing the general type of task the model is supposed to perform for all target properties."""

    REGRESSION = 'REGRESSION'
    SINGLECLASS = 'SINGLECLASS'
    MULTICLASS = 'MULTICLASS'
    MULTIOUTPUT_REGRESSION = 'MULTIOUTPUT_REGRESSION'
    MULTIOUTPUT_SINGLECLASS = 'MULTIOUTPUT_SINGLECLASS'
    MULTIOUTPUT_MULTICLASS = 'MULTIOUTPUT_MULTICLASS'
    MULTIOUTPUT_MIXED = 'MULTIOUTPUT_MIXED'

    def isClassification(self):
        """Check if the task is a classification task."""
        return self in [self.SINGLECLASS, self.MULTICLASS, self.MULTIOUTPUT_SINGLECLASS, self.MULTIOUTPUT_MULTICLASS]

    def isRegression(self):
        """Check if the task is a regression task.""""
        return self in [self.REGRESSION, self.MULTIOUTPUT_REGRESSION]

    def isMixed(self):
        """Check if the task is a mixed task.""""
        return self in [self.MULTIOUTPUT_MIXED]

    def isMultioutput(self):
        """Check if the task is a multioutput task.""""
        return self in [self.MULTIOUTPUT_REGRESSION, self.MULTIOUTPUT_SINGLECLASS,
                        self.MULTIOUTPUT_MULTICLASS, self.MULTIOUTPUT_MIXED]
