import numpy as np


class IntToZMapperMixin(object):

    def _int_from_z(self, z_value: float, atol=1.0e-6) -> int:
        if not hasattr(self, '_z_to_int'):
            self._z_to_int = dict()
            self._int_to_z = dict()

        result = None
        max_value = -1
        for z_test, int_val in self._z_to_int.items():
            d = np.abs(z_value-z_test)
            if int_val > max_value:
                max_value = int_val
            if d <= atol:
                result = int_val
        if result is None:
            new_val = max_value + 1
            self._z_to_int[z_value] = new_val
            result = new_val
            self._int_to_z[new_val] = z_value
        return result

    def _z_from_int(self, ii: int) -> float:
        return self._int_to_z[ii]
