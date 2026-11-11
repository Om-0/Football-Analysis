"""Basic smoke tests for environment and package installation."""

import pitchmind


def test_pitchmind_import():
    assert pitchmind.__version__ == "0.1.0"

