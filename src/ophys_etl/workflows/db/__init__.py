from airflow.providers.sqlite.hooks.sqlite import SqliteHook

engine = SqliteHook(sqlite_conn_id='ophys_workflow_db').get_sqlalchemy_engine()
