from pathlib import Path

import pytz

from ophys_etl.test_utils.workflow_utils import setup_app_config
from ophys_etl.workflows.app_config.slurm import SlurmSettings

setup_app_config(
    ophys_workflow_app_config_path=(
        Path(__file__).parent.parent.parent / "resources" / "config.yml"
    ),
    test_di_base_model_path=(
        Path(__file__).parent.parent.parent / "resources" / "di_model.h5"
    ),
)

import datetime  # noqa #402
import tempfile # noqa #402

from unittest.mock import patch, PropertyMock  # noqa #402

import pytest  # noqa #402

from ophys_etl.workflows.app_config.app_config import app_config  # noqa #E402
from ophys_etl.workflows.on_prem.slurm.slurm import Slurm  # noqa #402
from ophys_etl.workflows.on_prem.slurm.slurm import SlurmJob, SlurmState
from ophys_etl.workflows.ophys_experiment import (
    OphysExperiment,  # noqa #402
    OphysSession,
    Specimen, OphysContainer
)
from ophys_etl.workflows.pipeline_modules.motion_correction import (
    MotionCorrectionModule,
)  # noqa #402


class TestSlurmJob:
    @patch("requests.get")
    @pytest.mark.parametrize("has_job_started", (True, False))
    def test_from_job_id(self, get, has_job_started):
        class DummyResponse:
            status_code = 200

            @staticmethod
            def json():
                if has_job_started:
                    return {
                        "jobs": [
                            {
                                "state": {"current": SlurmState.COMPLETED},
                                'time': {
                                    'start': 1680557440,
                                    'end': 1680557518
                                }
                            }
                        ]
                    }
                else:
                    return {"jobs": []}

        get.return_value = DummyResponse

        job = SlurmJob.from_job_id(job_id="7353332")
        assert job.id == "7353332"

        if has_job_started:
            assert job.state == SlurmState.COMPLETED
            assert job.start == datetime.datetime(
                year=2023,
                month=4,
                day=3,
                hour=21,
                minute=30,
                second=40,
                tzinfo=pytz.UTC,
            )
            assert job.end == datetime.datetime(
                year=2023,
                month=4,
                day=3,
                hour=21,
                minute=31,
                second=58,
                tzinfo=pytz.UTC,
            )
        else:
            assert job.state is None
            assert job.start is None
            assert job.end is None

    @pytest.mark.parametrize(
        "job_state", (SlurmState.COMPLETED, SlurmState.RUNNING)
    )
    def test_is_done(self, job_state):
        job = SlurmJob(id="1", state=job_state)
        is_done = job.is_done()
        if job.state == SlurmState.COMPLETED:
            assert is_done
        else:
            assert not is_done

    @pytest.mark.parametrize(
        "job_state",
        (SlurmState.FAILED, SlurmState.TIMEOUT, SlurmState.COMPLETED),
    )
    def test_is_failed(self, job_state):
        job = SlurmJob(id="1", state=job_state)
        is_failed = job.is_failed()
        if job.state in (SlurmState.FAILED, SlurmState.TIMEOUT):
            assert is_failed
        else:
            assert not is_failed


class TestSlurm:
    def setup(self):
        self.tempdir = tempfile.TemporaryDirectory()
        raw_movie_path = Path(self.tempdir.name) / "mov.h5"

        with open(raw_movie_path, "w") as f:
            f.write("foo")

    @patch.object(
        Slurm, "_get_tmp_storage", wraps=lambda adjustment_factor: 55
    )
    @patch.object(MotionCorrectionModule, 'output_path',
                  new_callable=PropertyMock)
    def test_write_job_to_disk(self, 
                               mock_output_path,
                               _):
        mock_output_path.return_value = Path(self.tempdir.name)
        mod = MotionCorrectionModule(
            ophys_experiment=OphysExperiment(
                id=1,
                session=OphysSession(id=1, specimen=Specimen("1")),
                container=OphysContainer(id=1, specimen=Specimen("1")),
                specimen=Specimen(id="1"),
                storage_directory=Path(self.tempdir.name),
                raw_movie_filename=Path("mov.h5"),
                movie_frame_rate_hz=11.0,
                equipment_name='MESO.1',
                full_genotype="Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt"
            ),
            docker_tag=app_config.pipeline_steps.motion_correction.docker_tag,
        )
        slurm = Slurm(
            pipeline_module=mod,
            config=SlurmSettings(
                cpus_per_task=32,
                mem=250,
                time=300,
                request_additional_tmp_storage=True
            ),
            log_path=Path("log_path.log"),
        )
        slurm._write_job_to_disk(
            input_json="input.json",
            output_json="output.json"
        )
        with open(
            slurm._pipeline_module.output_path / "slurm_job.json"
        ) as f:
            job = f.read()

        with open(
            Path(__file__).parent / "test_data" / "expected_job.json"
        ) as f:
            expected_job = f.read()

        assert job == expected_job
    
    def teardown(self):
        self.tempdir.cleanup()
