"""
tasks

Created by: Martin Sicho
On: 28.11.22, 15:23
"""
from enum import Enum


class ModelTasks(Enum):
    REGRESSION = 'REGRESSION'
    SINGLECLASS = 'SINGLECLASS'
    MULTICLASS = 'MULTICLASS'

    def isClassification(self):
        return self in [self.SINGLECLASS, self.MULTICLASS]
