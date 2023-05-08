import datetime
from typing import List

import pandas as pd
from airflow.decorators import task
from airflow.models.dag import dag
from ophys_etl.workflows.utils.lims_utils import LIMSDB

from ophys_etl.workflows.db import engine
from sqlmodel import Session

from ophys_etl.workflows.utils.dag_utils import get_most_recent_run
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_step_runs import get_runs_completed_since
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


def _get_multiplane_experiments(ophys_experiment_ids: List[str]) -> List[str]:
    """Returns only those experiments from ophys_experiment_ids which are
    multiplane"""
    if len(ophys_experiment_ids) == 0:
        return []

    lims_db = LIMSDB()

    if len(ophys_experiment_ids) > 1:
        oe_ids_clause = f'oe.id in {tuple(ophys_experiment_ids)}'
    else:
        oe_ids_clause = f'oe.id = {ophys_experiment_ids[0]}'

    query = f'''
        SELECT oe.id as ophys_experiment_id, pg.group_order AS plane_group
        FROM  ophys_experiments oe
        JOIN ophys_sessions os ON oe.ophys_session_id = os.id
        JOIN  ophys_imaging_plane_groups pg
            ON pg.id = oe.ophys_imaging_plane_group_id
        WHERE os.id = (
            SELECT oe.ophys_session_id
            FROM ophys_experiments oe
            WHERE {oe_ids_clause}
        )
    '''
    res = lims_db.query(query=query)
    res = pd.DataFrame(res)
    plane_count = \
        res.groupby('ophys_experiment_id')['plane_group'].nunique()\
        .reset_index()\
        .rename(columns={'plane_group': 'plane_group_count'})
    multiplane = plane_count[plane_count[plane_count['plane_group_count'] > 1]]
    return multiplane['ophys_experiment_id'].tolist()


@dag(
    dag_id='decrosstalk_trigger',
    schedule='*/5 * * * *',  # every 5 minutes
    catchup=False,
    start_date=datetime.datetime.now()
)
def decrosstalk_trigger():
    """Checks for any ophys experiments that have completed segmentation
    since the last time this ran. If so, and
    1) this is a multiplane experiment and
    2) all other experiments from this session have also completed segmentation
    then we trigger decrosstalk for this session.
    """
    @task
    def trigger():
        last_run_datetime = get_most_recent_run(
            dag_id='decrosstalk_trigger')
        with Session(engine) as session:
            segmentation_runs = get_runs_completed_since(
                session=session,
                since=last_run_datetime,
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                workflow_step=WorkflowStepEnum.SEGMENTATION
            )
        ophys_experiment_ids = \
            [x.ophys_experiment_id for x in segmentation_runs]

        # 1. Filter to only multiplane experiments
        ophys_experiment_ids = _get_multiplane_experiments(
            ophys_experiment_ids=ophys_experiment_ids
        )




