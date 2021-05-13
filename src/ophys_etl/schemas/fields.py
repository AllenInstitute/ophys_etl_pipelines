from marshmallow import fields, ValidationError
import h5py
import argschema


class ExistingFile(argschema.fields.InputFile):
    pass


class H5InputFile(fields.Str):
    """
    H5InputFile is a subclass of :class:`marshmallow.fields.Str` which
    is a path to an h5 file location. The file must end with an extension
    of '.h5' or '.hdf5' and must be able to be opened by `h5py.File`.
    """
    def _validate(self, value):
        if not (str(value).endswith(".h5") or str(value).endswith(".hdf5")):
            raise ValidationError("H5 input file must have extension '.h5' "
                                  f"or '.hdf5'. Input file = {value}")
        try:
            with h5py.File(value, "r"):
                pass
        except OSError as e:
            raise ValidationError(f"Error occurred loading file {value}. "
                                  f"Underlying error: \nOSError: - {e}")


class ExistingH5File(H5InputFile):
    pass
