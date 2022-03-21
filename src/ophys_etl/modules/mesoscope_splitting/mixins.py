import numpy as np


class IntFromZMapperMixin(object):
    """
    This mixin defines methods to construct a mapping from
    floats to unique integer identifiers where floats are
    considered identical within some tolerance. The TIFF
    splitting classes use the methods provided to convert
    z-values which are floats into integers suitable for
    use as keys in dicts.
    """

    def _int_from_z(self,
                    z_value: float,
                    atol: float = 1.0e-6) -> int:
        """
        Convert a z_value into a unique integer, recording
        the mapping for reuse later, if necessary

        Parameters
        ----------
        z_value: float

        atol: float
           The absolute tolerance within which two floats
           are considered identical for the purposes of this
           method (if abs(f0-f1) <= atol, then it is considered
           that f0==f1)

           Note: if two recorded values are within atol of
           z_value, then the closest one is chosen.

        Returns
        -------
        int
           The unique integer associated with this z-value.
        """
        if not hasattr(self, '_int_from_z_lookup'):
            self._int_from_z_lookup = dict()
            self._z_from_int_lookup = dict()

        best_delta = None
        result = None
        max_value = -1
        for z_test, int_val in self._int_from_z_lookup.items():
            delta = np.abs(z_value-z_test)
            if int_val > max_value:
                max_value = int_val
            if delta <= atol:
                if best_delta is not None and delta > best_delta:
                    continue
                result = int_val
                best_delta = delta

        if result is None:
            new_val = max_value + 1
            self._int_from_z_lookup[z_value] = new_val
            result = new_val
            self._z_from_int_lookup[new_val] = z_value

        return result

    def _z_from_int(self, ii: int) -> float:
        """
        Return the float associated with a given integer
        in this lookup
        """
        return self._z_from_int_lookup[ii]
