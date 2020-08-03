import h5py
import pytest
from argschema import ArgSchemaParser, ArgSchema
from marshmallow import ValidationError
from ophys_etl.schemas.fields import H5InputFile


class H5InputTest(ArgSchema):
    input_file = H5InputFile(required=True, description="test")


@pytest.mark.parametrize(
    "schema", [(H5InputTest)]
)
def test_h5_input_file_errors_not_exist(schema, tmp_path):
    fp = tmp_path / "test.h5"
    with pytest.raises(ValidationError) as e:
        ArgSchemaParser(input_data={"input_file": str(fp)},
                        schema_type=H5InputTest,
                        args=[])
    assert "OSError" in str(e.value)


@pytest.mark.parametrize(
    "schema", [(H5InputTest)]
)
def test_h5_input_file_errors_not_h5(schema, tmp_path):
    fp = tmp_path / "starship.txt"
    fp.write_text("enterprise")
    with pytest.raises(ValidationError) as e:
        ArgSchemaParser(input_data={"input_file": str(fp)},
                        schema_type=H5InputTest,
                        args=[])
    assert ("H5 input file must have extension '.h5' or '.hdf5'"
            in str(e.value))


@pytest.mark.parametrize(
    "schema,filename", [
        (H5InputTest, "test.h5"),
        (H5InputTest, "test.hdf5")]
)
def test_h5_input_works(schema, filename, tmp_path):
    fp = tmp_path / "test.h5"
    with h5py.File(fp, "w"):
        pass
    parser = ArgSchemaParser(
        input_data={"input_file": str(fp)},
        schema_type=H5InputTest, args=[])
    assert parser.args["input_file"] == str(fp)
