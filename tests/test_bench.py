import asyncio

import numpy as np

from voxguard.backends.audio import mulaw_decode, mulaw_encode, telephony_channel
from voxguard.backends.mock import MockAgent, MockShadow
from voxguard.bench.dataset import generate_cases
from voxguard.bench.metrics import score_field
from voxguard.bench.runner import run_benchmark
from voxguard.bench.vad_sim import simulate
from voxguard.vad import EntityAwareEndpointer, FixedEndpointer


def test_score_field_outcomes():
    assert score_field("phone", "6789320288", "6789320288", "6789320288", "pass") == ("correct", "correct")
    assert score_field("phone", "6789320288", "6789320988", "6789320988", "confirm") == ("wrong", "caught")
    assert score_field("phone", "6789320288", "6789320988", "6789320288", "correct") == ("wrong", "corrected")
    assert score_field("phone", "6789320288", "6789320288", "6789320288", "confirm") == ("correct", "false_confirm")
    assert score_field("name", "Jon Smith", None, None, None) == ("missing", "missing")


def test_mock_benchmark_runs_and_improves():
    cases = generate_cases(25)
    rows, summ = asyncio.run(run_benchmark(cases, MockAgent(), MockShadow()))
    assert len(rows) == len(cases)
    allrow = summ[summ.kind == "all"].iloc[0]
    assert allrow.after_silent_error <= allrow.before_error


def test_vad_sim_entity_aware_cuts_in_less_than_fixed_500():
    cases = generate_cases(30)
    df = simulate(cases, {"f": FixedEndpointer(500), "e": EntityAwareEndpointer()})
    rates = df.groupby("endpointer").cut_in.mean()
    assert rates["e"] < rates["f"]


def test_telephony_channel_shape_and_mulaw_roundtrip():
    x = np.sin(np.linspace(0, 200 * np.pi, 16000)).astype(np.float32) * 0.5
    y, sr = telephony_channel(x, 16000, snr_db=30)
    assert sr == 8000 and abs(len(y) - 8000) < 5
    q = mulaw_decode(mulaw_encode(x))
    assert np.max(np.abs(q - x)) < 0.02
