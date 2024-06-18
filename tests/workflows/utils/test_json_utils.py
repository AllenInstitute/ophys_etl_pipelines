import tempfile
from dataclasses import dataclass
from enum import Enum
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

    def test_encode_enum(self):
        class Color(Enum):
            RED = 'RED'
            BLUE = 'BLUE'

        colors = [
            {'color': Color.RED},
            {'color': Color.BLUE}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir) / 'foo.json'
            with open(tmppath, 'w') as f:
                f.write(json.dumps(colors, cls=EnhancedJSONEncoder))

            with open(tmppath) as f:
                actual = json.load(f)

            assert actual == [{'color': 'RED'}, {'color': 'BLUE'}]

    def test_encode_dataclass_with_enum(self):
        class Color(Enum):
            RED = 'RED'
            BLUE = 'BLUE'

        @dataclass
        class Foo:
            color: Color

        foo = Foo(color=Color.BLUE)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir) / 'foo.json'
            with open(tmppath, 'w') as f:
                f.write(json.dumps(foo, cls=EnhancedJSONEncoder))

            with open(tmppath) as f:
                actual = json.load(f)

            assert actual == {'color': 'BLUE'}
