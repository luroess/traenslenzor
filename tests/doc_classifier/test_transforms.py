from __future__ import annotations

import pytest

from traenslenzor.doc_classifier.data_handling.transforms import TransformConfig


def test_transform_config_requires_override():
    cfg = TransformConfig()
    with pytest.raises(NotImplementedError):
        cfg.setup_target()
