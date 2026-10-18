import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "synthetic"))
from generator import generate_dataset

from fagi import FAGIDetector


@pytest.fixture(scope="module")
def sessions():
    return generate_dataset(n_per_org_benign=20, n_per_org_per_attack=5, seed=1)


class TestDetector:
    def test_predict_before_fit_raises(self, sessions):
        with pytest.raises(RuntimeError):
            FAGIDetector().predict(sessions)

    def test_fit_reports_computed_epsilon(self, sessions):
        det = FAGIDetector(batch_size=16).fit(sessions, epochs=2, verbose=False)
        assert det.epsilon is not None and np.isfinite(det.epsilon) and det.epsilon > 0
        assert det.train_info["sigma"] == 1.1

    def test_predict_shapes(self, sessions):
        det = FAGIDetector(use_dp=False, batch_size=16).fit(sessions, epochs=1, verbose=False)
        p = det.predict(sessions)
        pa = det.predict(sessions, adversarial=True)
        pr = det.predict_proba(sessions)
        assert p.shape == pa.shape == (len(sessions),)
        assert pr.shape == (len(sessions), 2) and np.allclose(pr.sum(1), 1, atol=1e-5)

    def test_save_load_roundtrip(self, sessions, tmp_path):
        det = FAGIDetector(use_dp=False, use_adversarial_training=False).fit(sessions, epochs=1, verbose=False)
        path = str(tmp_path / "m.pt")
        det.save(path)
        det2 = FAGIDetector(use_dp=False).load(path)
        assert np.array_equal(det.predict(sessions), det2.predict(sessions))
