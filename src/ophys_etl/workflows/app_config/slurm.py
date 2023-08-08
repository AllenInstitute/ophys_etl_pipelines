from typing import Optional

from pydantic import StrictInt, Field, StrictBool

from ophys_etl.workflows.utils.pydantic_model_utils import ImmutableBaseModel


class SlurmSettings(ImmutableBaseModel):
    cpus_per_task: StrictInt = 16
    mem: StrictInt = Field(description='Memory per node in GB', default=64)
    time: StrictInt = Field(description='Time limit in minutes', default=60)
    gpus: Optional[StrictInt] = Field(
        description='Number of GPUs',
        default=0
    )
    request_additional_tmp_storage: StrictBool = Field(
        default=False,
        description='If True, creates additional tmp storage'
    )
