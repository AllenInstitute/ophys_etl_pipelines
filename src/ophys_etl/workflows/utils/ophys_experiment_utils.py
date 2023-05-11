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
