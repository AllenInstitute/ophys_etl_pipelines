from typing import List, Dict

from ophys_etl.workflows.utils.lims_utils import LIMSDB


def get_session_experiment_id_map(
        ophys_experiment_ids: List[str]
) -> List[Dict]:
    """Get full list of experiment ids for each ophys session that each
    ophys_experiment_id belongs to"""
    lims_db = LIMSDB()

    if len(ophys_experiment_ids) == 0:
        oe_ids_clause = 'false'
    elif len(ophys_experiment_ids) > 1:
        oe_ids_clause = f'oe.id in {tuple(ophys_experiment_ids)}'
    else:
        oe_ids_clause = f'oe.id = {ophys_experiment_ids[0]}'

    query = f'''
        SELECT oe.id as ophys_experiment_id, os.id as ophys_session_id
        FROM  ophys_experiments oe
        JOIN ophys_sessions os ON oe.ophys_session_id = os.id
        WHERE os.id = (
            SELECT oe.ophys_session_id
            FROM ophys_experiments oe
            WHERE {oe_ids_clause}
        )
    '''
    res = lims_db.query(query=query)
    return res


def get_container_experiment_id_map(
    ophys_experiment_ids: List[str],
    exclude_failed_experiments: bool = True
) -> List[Dict]:
    """Get full list of experiment ids for each ophys container that each
    ophys_experiment_id belongs to

    Parameters
    ----------
    ophys_experiment_ids
        List of experiment ids to retrieve containers for
    exclude_failed_experiments
        Exclude any experiments not marked as "passed" or "qc"
    """
    lims_db = LIMSDB()

    if len(ophys_experiment_ids) == 0:
        oe_ids_clause = 'false'
    elif len(ophys_experiment_ids) > 1:
        oe_ids_clause = f'oe.id in {tuple(ophys_experiment_ids)}'
    else:
        oe_ids_clause = f'oe.id = {ophys_experiment_ids[0]}'

    query = f'''
        SELECT
            oevbec.visual_behavior_experiment_container_id as container_id,
            oe.id as ophys_experiment_id
        FROM  ophys_experiments_visual_behavior_experiment_containers oevbec
        JOIN ophys_experiments oe ON oe.id = oevbec.ophys_experiment_id
        WHERE oevbec.visual_behavior_experiment_container_id IN (
            SELECT oevbec.visual_behavior_experiment_container_id as container_id
            FROM  ophys_experiments_visual_behavior_experiment_containers oevbec
            JOIN ophys_experiments oe ON oe.id = oevbec.ophys_experiment_id
            WHERE {oe_ids_clause}
                {"AND oe.workflow_state in ('passed', 'qc')" if exclude_failed_experiments else ""}
    ''' # noqa E402
    res = lims_db.query(query=query)
    return res
