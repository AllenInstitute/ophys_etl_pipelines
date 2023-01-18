import tempfile
from dataclasses import dataclass
from pathlib import Path

import json

from ophys_etl.workflows.utils.json_utils import EnhancedJSONEncoder


class TestEnhancedJsonEncoder:
    def test_encode_data_class(self):
        @dataclass
        class Foo:
            x: int
            y: int
        foo = Foo(x=1, y=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir) / 'foo.json'
            with open(tmppath, 'w') as f:
                f.write(json.dumps(foo, cls=EnhancedJSONEncoder))

            with open(tmppath) as f:
                actual = json.load(f)

            assert actual == {'x': 1, 'y': 2}

    def test_encode_path(self):
        x = {
            'path': Path('/foo')
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir) / 'foo.json'
            with open(tmppath, 'w') as f:
                f.write(json.dumps(x, cls=EnhancedJSONEncoder))

            with open(tmppath) as f:
                actual = json.load(f)

            assert actual == {'path': '/foo'}
