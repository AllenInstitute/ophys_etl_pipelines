from enum import Enum


class WorkflowName(Enum):
    """Available workflow types"""
    OPHYS_PROCESSING = 'OPHYS_PROCESSING'
    ROI_CLASSIFIER_TRAINING = 'ROI_CLASSIFIER_TRAINING'
