from pathlib import Path

from ophys_etl.test_utils.workflow_utils import setup_app_config

setup_app_config(
    ophys_workflow_app_config_path=(
            Path(__file__).parent.parent.parent / 'resources' / 'config.yml'),
    test_di_base_model_path=(
            Path(__file__).parent.parent.parent / 'resources' / 'di_model.h5')
)

import datetime # noqa #402
from unittest.mock import patch # noqa #402

import pytest   # noqa #402
from ophys_etl.workflows.ophys_experiment import OphysExperiment, \
    OphysSession, Specimen  # noqa #402

from ophys_etl.workflows.on_prem.slurm.slurm import SlurmJob, SlurmState, \
    Slurm, _parse_job_id_from_sbatch_output # noqa #402
from ophys_etl.workflows.pipeline_modules.motion_correction import \
    MotionCorrectionModule  # noqa #402


class TestSlurmJob:
    @pytest.mark.parametrize('has_job_started', (True, False))
    def test_from_job_id(self, has_job_started):
        meta = 'job_meta.txt' if has_job_started else \
            'job_meta_not_started.txt'
        with open(Path(__file__).parent / 'test_data' / meta) as f:
            dummy_job_meta = f.read()

        with patch.object(SlurmJob, '_get_job_meta',
                          wraps=lambda job_id: dummy_job_meta):
            job = SlurmJob.from_job_id(job_id='7353332')
        assert job.id == '7353332'

        if has_job_started:
            assert job.state == SlurmState.COMPLETED
            assert job.start == datetime.datetime(
                year=2023, month=1, day=18, hour=15, minute=24, second=52)
            assert job.end == datetime.datetime(
                year=2023, month=1, day=18, hour=15, minute=24, second=57)
        else:
            assert job.state is None
            assert job.start is None
            assert job.end is None

    @pytest.mark.parametrize('valid', (True, False))
    def test_try_parse_datetime(self, valid):
        if valid:
            assert SlurmJob._try_parse_datetime('2023-01-18T15:24:57') == \
                   datetime.datetime(
                       year=2023, month=1, day=18, hour=15, minute=24,
                       second=57)
        else:
            assert SlurmJob._try_parse_datetime('') is None

    @pytest.mark.parametrize('job_state',
                             (SlurmState.COMPLETED, SlurmState.RUNNING))
    def test_is_done(self, job_state):
        job = SlurmJob(id='1', state=job_state)
        is_done = job.is_done()
        if job.state == SlurmState.COMPLETED:
            assert is_done
        else:
            assert not is_done

    @pytest.mark.parametrize('job_state',
                             (SlurmState.FAILED, SlurmState.TIMEOUT,
                              SlurmState.COMPLETED))
    def test_is_failed(self, job_state):
        job = SlurmJob(id='1', state=job_state)
        is_failed = job.is_failed()
        if job.state in (SlurmState.FAILED, SlurmState.TIMEOUT):
            assert is_failed
        else:
            assert not is_failed


class TestSlurm:
    @classmethod
    def setup_class(cls):
        mod = MotionCorrectionModule(
            ophys_experiment=OphysExperiment(
                id='1',
                session=OphysSession(id='1'),
                specimen=Specimen(id='1'),
                storage_directory=Path('/foo'),
                raw_movie_filename=Path('mov.h5'),
                movie_frame_rate_hz=11.0
            )
        )
        with patch.object(Slurm, '_get_tmp_storage',
                          wraps=lambda adjustment_factor: 55):
            cls._slurm = Slurm(
                ophys_experiment_id='1',
                pipeline_module=mod,
                config_path=Path(__file__).parent / 'test_data' /
                'slurm_config.yml',
                log_path=Path('log_path.log')
            )

    def test_get_slurm_script_headers(self):
        with open(Path(__file__).parent / 'test_data' /
                  'expected_headers.txt') as f:
            expected_headers = f.read()
        assert self._slurm._slurm_headers == expected_headers

    def test_write_script(self):
        self._slurm._write_script(
            input_json='input.json', output_json='output.json')
        with open(self._slurm._pipeline_module.output_path /
                  f'{self._slurm._ophys_experiment_id}.slurm') as f:
            script = f.read()

        with open(Path(__file__).parent / 'test_data' /
                  'expected_command.txt') as f:
            expected_cmd = f.read()

        cmd = script.splitlines()[-2]
        assert cmd == expected_cmd


def test_parse_job_id_from_sbatch_output_valid():
    output = 'submitted batch job with id 1234'
    job_id = _parse_job_id_from_sbatch_output(output=output)
    assert job_id == '1234'


@pytest.mark.parametrize('output', ('', '1234 1234'))
def test_parse_job_id_from_sbatch_output_invalid(output):
    with pytest.raises(RuntimeError):
        _parse_job_id_from_sbatch_output(output=output)
