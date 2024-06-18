from sqlalchemy import event

from ophys_etl.workflows.app_config.app_config import app_config
from sqlmodel import create_engine


def _fk_pragma_on_connect(dbapi_con, con_record):
    """Needed for sqlite to enforce foreign key constraint"""
    dbapi_con.execute('pragma foreign_keys=ON')


def get_engine(
        db_conn: str = app_config.app_db.conn_string
):
    engine = create_engine(db_conn)
    if db_conn.startswith('sqlite'):
        event.listen(engine, 'connect', _fk_pragma_on_connect)
    return engine


engine = get_engine()
