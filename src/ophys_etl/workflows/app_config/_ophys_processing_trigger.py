"""Config for ophys processing trigger"""
from typing import Tuple

from pydantic import Field

from ophys_etl.workflows.utils.pydantic_model_utils import ImmutableBaseModel


class OphysProcessingTrigger(ImmutableBaseModel):
    lims_trigger_queues: Tuple[str] = Field(
        default=('MESOSCOPE_FILE_SPLITTING_QUEUE',),
        description='List of LIMS queues that will trigger ophys processing'
    )
