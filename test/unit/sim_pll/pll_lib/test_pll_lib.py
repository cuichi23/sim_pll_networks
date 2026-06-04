"""Unit tests for module-level helpers in sim_pll.pll_lib."""
import numpy as np
import pytest

from sim_pll.pll_lib import get_from_value_or_list


class TestGetFromValueOrList:

    def test_none_passthrough(self):
        assert get_from_value_or_list(0, None, 3) is None

    def test_scalar_broadcast(self):
        assert get_from_value_or_list(2, 1.5, 3) == 1.5
        assert get_from_value_or_list(0, 4, 3) == 4

    def test_list_indexed_by_pll_id(self):
        values = [10.0, 20.0, 30.0]
        assert get_from_value_or_list(0, values, 3) == 10.0
        assert get_from_value_or_list(2, values, 3) == 30.0

    def test_ndarray_indexed_by_pll_id(self):
        values = np.array([7.0, 8.0, 9.0])
        assert get_from_value_or_list(1, values, 3) == 8.0

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError):
            get_from_value_or_list(0, {'bad': 'type'}, 3)
