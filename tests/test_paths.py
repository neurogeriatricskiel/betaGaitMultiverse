"""Per-dataset directories resolve and do not collide."""

from src.config import DATASETS
from src.paths import get_dataset_dirs


def test_dataset_dirs_differ():
    st, jc = get_dataset_dirs("stepup"), get_dataset_dirs("jacobsen")
    assert set(st) == set(jc)
    for key in st:
        assert st[key] != jc[key]
        assert st[key].parent == DATASETS["stepup"]["root"]
        assert jc[key].parent == DATASETS["jacobsen"]["root"]


def test_expected_keys():
    keys = set(get_dataset_dirs("stepup"))
    assert {"raw", "gait_events", "prep", "clean", "gaitepochs", "ersp", "qc"} <= keys
