"""Interface to the LIMS database"""
from typing import List, Dict

from sqlalchemy import create_engine

from ophys_etl.workflows.app_config.app_config import app_config
from sqlalchemy.engine import URL, Engine


class LIMSDB:
    def __init__(self):
        self._engine = self._get_lims_engine()

    @staticmethod
    def _get_lims_engine() -> Engine:
        """Get sqlalchemy engine for lims DB"""
        url = URL.create(
            drivername='postgresql+pg8000',
            username=app_config.lims_db.username.get_secret_value(),
            password=app_config.lims_db.password.get_secret_value(),
            host='limsdb2.corp.alleninstitute.org',
            database='lims2',
        )
        engine = create_engine(url=url)
        return engine

    def query(self, query: str) -> List[Dict]:
        """Queries DB and returns result"""
        with self._engine.connect() as conn:
            res = conn.execute(query)
            columns = res.keys()
            values = res.all()

        res = [dict(zip(columns, x)) for x in values]
        return res
