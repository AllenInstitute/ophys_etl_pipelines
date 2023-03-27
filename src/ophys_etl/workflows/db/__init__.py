from sqlalchemy import event

from ophys_etl.workflows.app_config.app_config import app_config
from sqlmodel import create_engine


def _fk_pragma_on_connect(dbapi_con, con_record):
    """Needed for sqlite to enforce foreign key constraint"""
    dbapi_con.execute('pragma foreign_keys=ON')


def get_engine(
        db_conn: str = app_config.app_db.conn_string
):
    if db_conn.startswith('sqlite'):
        engine = create_engine(db_conn)
        event.listen(engine, 'connect', _fk_pragma_on_connect)
    else:
        raise NotImplementedError('only sqlite currently supported')
    return engine


engine = get_engine()
