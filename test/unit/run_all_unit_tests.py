"""Thin wrapper that runs the unit-test suite via pytest.

The test suite was migrated from unittest stubs to pytest. This script remains so
that ``python test/unit/run_all_unit_tests.py`` keeps working; it simply delegates
to pytest on the ``test/unit`` directory.
"""
import os
import sys

import pytest


def main() -> int:
    """Run all unit tests under test/unit and return pytest's exit code."""
    unit_dir = os.path.dirname(os.path.abspath(__file__))
    return pytest.main([unit_dir, "-q"])


if __name__ == "__main__":
    sys.exit(main())
