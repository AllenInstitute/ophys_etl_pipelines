"""Slurm interface"""
import os
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import logging

import yaml
from paramiko import SSHClient
from pydantic import StrictInt, StrictStr, Field
from simple_slurm import Slurm as SimpleSlurm

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.utils.pydantic_model_utils import ImmutableBaseModel

logger = logging.getLogger('airflow.task')


class _SlurmSettings(ImmutableBaseModel):
    cpus_per_task: StrictInt
    mem: StrictInt
    time: StrictStr = Field(description='Timeout, use format H:M:S')
    gpus: Optional[StrictInt] = Field(
        description='Number of GPUs',
        default=0
    )


def _exec_slurm_command(command: str) -> str:
    """Execute command over ssh on hpc-login"""
    with SSHClient() as client:
        client.load_system_host_keys()
        client.connect('hpc-login')
        _, stdout, stderr = client.exec_command(command=command)
        stderr = stderr.read()
        if stderr:
            raise RuntimeError(stderr)
        stdout = bytes.decode(stdout.read())
    return stdout


class SlurmState(Enum):
    """States that a slurm job can be in

    Source: https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES
    """
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    RUNNING = 'RUNNING'
    PENDING = 'PENDING'
    BOOT_FAIL = 'BOOT_FAIL'
    CANCELLED = 'CANCELLED'
    CONFIGURING = 'CONFIGURING'
    COMPLETING = 'COMPLETING'
    DEADLINE = 'DEADLINE'
    NODE_FAIL = 'NODE_FAIL'
    OUT_OF_MEMORY = 'OUT_OF_MEMORY'
    PREEMPTED = 'PREEMPTED'
    RESV_DEL_HOLD = 'RESV_DEL_HOLD'
    REQUEUE_FED = 'REQUEUE_FED'
    REQUEUE_HOLD = 'REQUEUE_HOLD'
    REQUEUED = 'REQUEUED'
    RESIZING = 'RESIZING'
    REVOKED = 'REVOKED'
    SIGNALING = 'SIGNALING'
    SPECIAL_EXIT = 'SPECIAL_EXIT'
    STAGE_OUT = 'STAGE_OUT'
    STOPPED = 'STOPPED'
    SUSPENDED = 'SUSPENDED'
    TIMEOUT = 'TIMEOUT'
    CANCELLED_plus = 'CANCELLED+'


class SlurmJobFailedException(Exception):
    """The slurm job failed"""
    pass


@dataclass
class SlurmJob:
    """A job that has been submitted to slurm

    Attributes:

    - :class:`id`: Slurm job id
    - :class:`state`: slurm job state. None if not started yet
    - :class:`start`: start time of job
    - :class:`end`: end time of job
    """
    id: str
    state: Optional[SlurmState] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    @classmethod
    def from_job_id(
            cls,
            job_id: str,
    ) -> "SlurmJob":
        """Parses job metadata from sacct command
        The output should look like

        JobID             State               Start                 End
        ------------ ---------- ------------------- -------------------
        7353332       COMPLETED 2023-01-18T15:24:52 2023-01-18T15:24:57
        7353332.bat+  COMPLETED 2023-01-18T15:24:52 2023-01-18T15:24:57
        7353332.ext+  COMPLETED 2023-01-18T15:24:52 2023-01-18T15:24:58

        Returns
        -------
        If job id `job_id` found, returns `SlurmJob` instance

        """
        output = cls._get_job_meta(job_id=job_id)
        lines = output.splitlines()
        lines = lines[2:]
        lines = [line.split() for line in lines]
        job_metas = [{
            'job_id': job_id,
            'state': state,
            'start': cls._try_parse_datetime(datetime_string=start),
            'end': cls._try_parse_datetime(datetime_string=end)
        } for job_id, state, start, end in lines]
        for job_meta in job_metas:
            if job_meta['job_id'] == job_id:
                job = cls(
                    id=job_meta['job_id'],
                    state=SlurmState(job_meta['state']),
                    start=job_meta['start'],
                    end=job_meta['end']
                )
                return job

        # If we got here it means the job has not started yet
        return cls(
            id=job_id
        )

    @staticmethod
    def _get_job_meta(job_id) -> str:
        stdout = _exec_slurm_command(
            command=f'sacct --format JobID,State,Start,End --job={job_id}')
        return stdout

    @staticmethod
    def _try_parse_datetime(datetime_string: str) -> Optional[datetime]:
        try:
            res = datetime.strptime(datetime_string, '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            res = None
        return res

    def is_done(self) -> bool:
        """Whether the job is done"""
        return self.state == SlurmState.COMPLETED

    def is_failed(self) -> bool:
        """Whether the job failed"""
        failed_states = [
            SlurmState.FAILED,
            SlurmState.CANCELLED_plus,
            SlurmState.TIMEOUT,
            SlurmState.CANCELLED,
            SlurmState.OUT_OF_MEMORY,
        ]
        return self.state in failed_states


class Slurm:
    """Wrapper around slurm"""
    def __init__(
        self,
        ophys_experiment_id: str,
        pipeline_module: PipelineModule,
        config_path: Path,
        log_path: Path,
        tmp_storage_adjustment_factor: int = 3
    ):
        """
        Parameters
        ----------
        ophys_experiment_id`
            identifier for ophys experiment
        pipeline_module
            `PipelineModule` instance
        config_path
            Path to slurm settings
        log_path
            Where to write slurm job logs to
        tmp_storage_adjustment_factor
            Multiplies the ophys experiment file size by this amount to give
            some breathing room to the amount of tmp storage to reserve
        """
        self._pipeline_module = pipeline_module
        self._ophys_experiment_id = ophys_experiment_id
        self._job: Optional[SlurmJob] = None
        self._slurm_settings = read_config(config_path=config_path)

        os.makedirs(log_path.parent, exist_ok=True)

        self._slurm_headers = self._get_slurm_script_headers(
            job_name=f'{pipeline_module.queue_name}_{ophys_experiment_id}',
            log_path=log_path,
            tmp_storage_adjustment_factor=tmp_storage_adjustment_factor
        )

    @property
    def job(self) -> SlurmJob:
        return self._job

    def _write_script(
            self,
            *args,
            **kwargs
    ) -> Path:
        """
        Adds cmd to slurm headers, and writes script to disk

        Parameters
        ----------
        args
            positional args to pass to command
        kwargs
            keyword args to pass to command

        Returns
        -------
        Path to slurm script
        """
        args = ' '.join([f'{x}' for x in args])
        kwargs = ' '.join([f'--{k} {v}' for k, v in kwargs.items()])
        if self._pipeline_module.debug:
            cmd = f'{self._pipeline_module.executable} {args} {kwargs}'
        else:
            docker_tag = self._pipeline_module.docker_tag
            singularity_username = \
                app_config.singularity.username.get_secret_value()
            singularity_password = \
                app_config.singularity.password.get_secret_value()

            request_gpu = self._slurm_settings.gpus > 0
            cmd = f'''
# Adds mksquashfs (needed for singularity) to $PATH
source /etc/profile

export SINGULARITY_DOCKER_USERNAME={singularity_username}
export SINGULARITY_DOCKER_PASSWORD={singularity_password}

SINGULARITY_TMPDIR=/scratch/fast/${{SLURM_JOB_ID}} singularity run \
    --bind /allen:/allen,/scratch/fast/${{SLURM_JOB_ID}}:/tmp \
    {"--nv" if request_gpu else ""} \
    docker://alleninstitutepika/ophys_etl_pipelines:{docker_tag} \
    /envs/ophys_etl/bin/python -m {self._pipeline_module.executable} {args} \
    {kwargs}
            '''
        slurm_script = f'{self._slurm_headers}\n\n{cmd}'

        out = self._pipeline_module.output_path / \
            f'{self._ophys_experiment_id}.slurm'

        with open(out, 'w') as f:
            f.write(slurm_script)
        logger.info(f'Wrote slurm script to {out}')
        return Path(out)

    def submit_job(
        self,
        *args,
        **kwargs
    ) -> str:
        """
        Submits batch job to slurm

        Parameters
        ----------
        args
            positional args to pass to command
        kwargs
            keyword args to pass to command

        Returns
        -------
        slurm job id
        """
        script_path = self._write_script(*args, **kwargs)
        stdout = _exec_slurm_command(command=f'sbatch {script_path}')
        job_id = _parse_job_id_from_sbatch_output(
            output=stdout)
        job = SlurmJob.from_job_id(job_id=job_id)
        logger.info(stdout)
        self._job = job
        return job_id

    def _get_slurm_script_headers(
        self,
        job_name: str,
        log_path: Path,
        tmp_storage_adjustment_factor: int = 3
    ) -> str:
        """

        Parameters
        ----------
        job_name
            slurm job name
        log_path
            Where to write slurm job logs to
        tmp_storage_adjustment_factor
            Multiplies the ophys experiment file size by this amount to give
            some breathing room to the amount of tmp storage to reserve
        Returns
        -------
        The slurm script headers
        """
        if self._pipeline_module.debug:
            tmp = 0
        else:
            tmp = self._get_tmp_storage(
                adjustment_factor=tmp_storage_adjustment_factor)
        cpus_per_task = \
            1 if self._pipeline_module.debug else \
            self._slurm_settings.cpus_per_task
        mem = \
            1 if self._pipeline_module.debug else self._slurm_settings.mem
        time = \
            '00:10:00' if self._pipeline_module.debug else \
            self._slurm_settings.time
        gpus = \
            0 if self._pipeline_module.debug else \
            self._slurm_settings.gpus
        s = SimpleSlurm(
            partition='braintv',
            qos='production',
            nodes=1,
            cpus_per_task=cpus_per_task,
            gpus=gpus,
            mem=f'{mem}G',
            time=time,
            job_name=job_name,
            mail_type='NONE',
            tmp=f'{tmp}G',
            out=log_path
        )
        return str(s)

    def _get_tmp_storage(
        self,
        adjustment_factor: int = 3
    ) -> int:
        """
        Gets the amount of temporary storage to reserve

        Parameters
        ----------
        adjustment_factor
            Multiplies the ophys experiment file size by this amount to give
            some breathing room

        Returns
        -------
        Amount of file size to reserve in GB
        """
        storage_directory = \
            self._pipeline_module.ophys_experiment.storage_directory
        raw_movie_filename =\
            self._pipeline_module.ophys_experiment.raw_movie_filename
        file_size = (storage_directory / raw_movie_filename).stat().st_size
        return int(file_size / (1024 ** 3) * adjustment_factor)


def read_config(config_path: Path) -> _SlurmSettings:
    """
    Reads and validates slurm settings

    Parameters
    ----------
    config_path
        Path to slurm settings config

    Raises
    ------
    `ValidationError` if config is invalid

    Returns
    -------
    _SlurmSettings
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = _SlurmSettings(**config)
    return config


def _parse_job_id_from_sbatch_output(output: str) -> str:
    match = re.findall(r'\d+', output)
    if len(match) == 0:
        raise RuntimeError('Could not parse job id from sbatch command')
    if len(match) > 1:
        raise RuntimeError(f'Got unexpected output from sbatch command: '
                           f'{output}')
    return match[0]
