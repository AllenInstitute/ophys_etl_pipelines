import datetime
from pathlib import Path
from typing import Type, Optional, Dict, Any

from airflow.decorators import task_group
from airflow.models import Param
from airflow.models.dag import dag

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.db.db_utils import enable_fk_if_sqlite
from ophys_etl.workflows.tasks import save_job_run_to_db
from ophys_etl.workflows.on_prem.tasks import submit_job, \
    wait_for_job_to_finish
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.pipeline_modules.denoising.denoising_finetuning \
    import \
    DenoisingFinetuningModule
from ophys_etl.workflows.pipeline_modules.denoising.denoising_inference \
    import \
    DenoisingInferenceModule
from ophys_etl.workflows.pipeline_modules.motion_correction import \
    MotionCorrectionModule


def _run_workflow_step(
    slurm_config_filename: str,
    module: Type[PipelineModule],
    workflow_step_name: str,
    docker_tag: str,
    module_kwargs: Optional[Dict] = None
) -> Any:
    """
    Runs a single workflow step

    Parameters
    ----------
    slurm_config_filename
        Slurm settings filename
    module
        What module to run
    workflow_step_name
        Workflow step name
    docker_tag
        What docker tag to use
    module_kwargs
        kwargs to send to module

    Returns
    -------
    Dictionary mapping WellKnownFileType to OutputFile, but actually an
    airflow.models.XCom until pulled in a task
    """
    slurm_config = (Path(__file__).parent.parent / 'slurm' / 'configs' /
                    slurm_config_filename)
    job_submit_res = submit_job(
        module=module,
        config_path=str(slurm_config),
        docker_tag=docker_tag,
        module_kwargs=module_kwargs
    )

    job_finish_res = wait_for_job_to_finish(
        timeout=app_config.job_timeout
    )(
        job_id=job_submit_res['job_id'],
        storage_directory=job_submit_res['storage_directory'],
        module_outputs=(
            job_submit_res['module_outputs'])

    )
    module_outputs = save_job_run_to_db(
        workflow_step_name=workflow_step_name,
        job_finish_res=job_finish_res
    )

    return module_outputs


@dag(
    dag_id='ophys_processing',
    schedule=None,
    catchup=False,
    start_date=datetime.datetime.now(),
    # For required arguments, need to pass default of None due to
    # https://github.com/apache/airflow/issues/28940
    # This makes the argument optional. Unfortunately it is impossible to
    # make it required.
    # TODO once fixed, remove default
    params={
        'ophys_experiment_id': Param(
            description='identifier for ophys experiment',
            default=None
        ),
        'debug': Param(
            description='If True, will not actually run the modules, but '
                        'will run a dummy command instead to test the '
                        'workflow',
            default=False
        ),
        'prevent_file_overwrites': Param(
            description='If True, will fail job run if a file output by '
                        'module already exists',
            default=True
        )
    }
)
def ophys_processing():
    @task_group
    def motion_correction():
        """Motion correct raw ophys movie"""
        module_outputs = _run_workflow_step(
            slurm_config_filename='motion_correction.yml',
            module=MotionCorrectionModule,
            workflow_step_name='motion_correction',
            docker_tag=app_config.pipeline_steps.motion_correction.docker_tag
        )
        return module_outputs['MotionCorrectedImageStack']

    @task_group
    def denoising(
        motion_corrected_ophys_movie_file
    ):
        @task_group
        def denoising_finetuning(motion_corrected_ophys_movie_file):
            """Finetune deepinterpolation model on a single ophys movie"""
            module_outputs = _run_workflow_step(
                slurm_config_filename='denoising_finetuning.yml',
                module=DenoisingFinetuningModule,
                workflow_step_name='denoising_finetuning',
                docker_tag=app_config.pipeline_steps.denoising.docker_tag,
                module_kwargs={
                    'motion_corrected_ophys_movie_file':
                        motion_corrected_ophys_movie_file
                }
            )
            return module_outputs['DeepInterpolationFinetunedModel']

        @task_group
        def denoising_inference(
                motion_corrected_ophys_movie_file,
                trained_denoising_model_file
        ):
            """Runs denoising inference on a single ophys movie"""
            module_outputs = _run_workflow_step(
                slurm_config_filename='denoising_inference.yml',
                module=DenoisingInferenceModule,
                workflow_step_name='denoising_inference',
                docker_tag=app_config.pipeline_steps.denoising.docker_tag,
                module_kwargs={
                    'motion_corrected_ophys_movie_file':
                        motion_corrected_ophys_movie_file,
                    'trained_denoising_model_file':
                        trained_denoising_model_file
                }
            )
            return module_outputs['DeepInterpolationDenoisedOphysMovie']

        trained_denoising_model_file = denoising_finetuning(
            motion_corrected_ophys_movie_file=(
                motion_corrected_ophys_movie_file))
        denoised_movie = denoising_inference(
            motion_corrected_ophys_movie_file=(
                motion_corrected_ophys_movie_file),
            trained_denoising_model_file=trained_denoising_model_file)
        return denoised_movie

    motion_corrected_ophys_movie_file = motion_correction()
    denoising(
        motion_corrected_ophys_movie_file=motion_corrected_ophys_movie_file)


enable_fk_if_sqlite()

ophys_processing()
