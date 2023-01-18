"""Script to create and populate tables"""
from typing import Dict

import argschema
from sqlalchemy import create_engine, event
from sqlmodel import SQLModel, Session

from ophys_etl.workflows.db.db_utils import fk_pragma_on_connect
from ophys_etl.workflows.db.schemas import Workflow, WorkflowStep, \
    WellKnownFileType


class InitializeDBSchema(argschema.ArgSchema):
    db_url = argschema.fields.String(
        required=True,
        description='db url'
    )


def create_db_and_tables(engine):
    """Creates db and empty tables"""
    SQLModel.metadata.create_all(engine)


def _create_workflow(session):
    """Adds Workflow"""
    workflow = Workflow(name='ophys_processing')
    session.add(workflow)
    session.commit()
    return workflow


def _create_workflow_steps(session, workflow: Workflow):
    """Adds workflow steps"""
    workflow_steps = {
        'motion_correction': WorkflowStep(
            name='motion_correction',
            workflow_id=workflow.id
        ),
        'denoising_finetuning': WorkflowStep(
            name='denoising_finetuning',
            workflow_id=workflow.id
        ),
        'denoising_inference': WorkflowStep(
            name='denoising_inference',
            workflow_id=workflow.id
        ),
        'segmentation': WorkflowStep(
            name='segmentation',
            workflow_id=workflow.id
        ),
        'roi_classification': WorkflowStep(
            name='roi_classification',
            workflow_id=workflow.id
        ),
        'trace_extraction': WorkflowStep(
            name='trace_extraction',
            workflow_id=workflow.id
        )
    }
    for workflow_step in workflow_steps.values():
        session.add(workflow_step)
    session.commit()
    return workflow_steps


def _create_motion_correction_well_known_file_types(
    session,
    workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name='OphysMaxIntImage',
            workflow_step_id=workflow_steps['motion_correction'].id
        ),
        WellKnownFileType(
            name='OphysAverageIntensityProjectionImage',
            workflow_step_id=workflow_steps['motion_correction'].id
        ),
        WellKnownFileType(
            name='OphysRegistrationSummaryImage',
            workflow_step_id=workflow_steps['motion_correction'].id
        ),
        WellKnownFileType(
            name='MotionCorrectedImageStack',
            workflow_step_id=workflow_steps['motion_correction'].id
        ),
        WellKnownFileType(
            name='OphysMotionXyOffsetData',
            workflow_step_id=workflow_steps['motion_correction'].id
        ),
        WellKnownFileType(
            name='OphysMotionPreview',
            workflow_step_id=workflow_steps['motion_correction'].id
        )
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _create_denoising_well_known_file_types(
    session,
    workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name='DeepInterpolationFinetunedModel',
            workflow_step_id=workflow_steps['denoising_finetuning'].id
        ),
        WellKnownFileType(
            name='DeepInterpolationDenoisedOphysMovie',
            workflow_step_id=workflow_steps['denoising_inference'].id
        )
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _create_well_known_file_types(
        session,
        workflow_steps: Dict[str, WorkflowStep]
):
    """Adds well known file types"""
    _create_motion_correction_well_known_file_types(
        session=session,
        workflow_steps=workflow_steps
    )
    _create_denoising_well_known_file_types(
        session=session,
        workflow_steps=workflow_steps
    )
    session.commit()


def _populate_db(engine):
    """Populates tables"""
    with Session(engine) as session:
        workflow = _create_workflow(session=session)
        workflow_steps = _create_workflow_steps(
            session=session,
            workflow=workflow)
        _create_well_known_file_types(
            session=session,
            workflow_steps=workflow_steps)


class IntializeDBRunner(argschema.ArgSchemaParser):
    default_schema = InitializeDBSchema

    def run(self):
        engine = create_engine(self.args['db_url'])
        if self.args['db_url'].startswith('sqlite'):
            # enable foreign key constraint
            event.listen(engine, 'connect', fk_pragma_on_connect)

        create_db_and_tables(engine)
        _populate_db(engine)


if __name__ == "__main__":
    init_db = IntializeDBRunner()
    init_db.run()
