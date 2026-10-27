"""The GEDAI gate uses only the rows with role == primary (scripts/qc/qc_gedai_gate.py)."""

import importlib.util
import io
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

# Synthetic stepup_gedai_gate.tsv: the superseded and sensitivity rows carry
# values far from the primary ones, so any leak into the gate changes it.
TSV = """subject\trole\tbroadband_multiplier\tsegment\tfit\twelch_average\troi_beta_removed_pct
S1\tsuperseded\t3\twhole\twhole\tmean\t96.0
S1\tprimary\t3\twalking_trim\twhole\tmedian\t10.0
S1\tsensitivity\t3\twalking_trim\tseparate\tmedian\t80.0
S1\tsensitivity\t6\twalking_trim\twhole\tmedian\t70.0
S9\tsuperseded\t3\twhole\twhole\tmean\t99.0
S9\tprimary\t3\twalking_trim\twhole\tmedian\t20.0
S9\tsensitivity\t3\tstanding_trim\twhole\tmedian\t1.0
S2\tprimary\t3\twalking_trim\twhole\tmedian\t90.0
"""


@pytest.fixture
def gate(monkeypatch):
    """Load the script as a module without running it."""
    monkeypatch.setenv("BETAGAIT_DATASET", "stepup")
    path = ROOT / "scripts" / "qc" / "qc_gedai_gate.py"
    spec = importlib.util.spec_from_file_location("qc_gedai_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_uses_only_primary_rows(gate):
    table = pd.read_csv(io.StringIO(TSV), sep="\t")
    # mean of the S1 and S9 primary rows; S2 is not a gate subject
    assert gate.gate_value(table, ["S1", "S9"]) == pytest.approx(15.0)


def test_gate_needs_one_primary_row_per_subject(gate):
    table = pd.read_csv(io.StringIO(TSV), sep="\t")
    with pytest.raises(ValueError):
        gate.gate_value(table, ["S1", "S9", "S14"])


def test_assign_role(gate):
    table = pd.read_csv(io.StringIO(TSV), sep="\t").drop(columns="role")
    expected = pd.read_csv(io.StringIO(TSV), sep="\t")["role"]
    assert list(gate.assign_role(table)) == list(expected)
