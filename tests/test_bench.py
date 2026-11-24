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


def test_expand_accents_and_bootstrap():
    from voxguard.bench.dataset import expand_accents
    from voxguard.bench.metrics import bootstrap_ci
    cases = expand_accents(generate_cases(5), ["us", "indian"])
    assert len(cases) == 40 and {c.accent for c in cases} == {"us", "indian"}
    assert expand_accents(cases, ["us"]) == cases  # already expanded cases are kept as-is
    rows, _ = asyncio.run(run_benchmark(cases, MockAgent(), MockShadow()))
    assert set(rows["accent"]) == {"us", "indian"}
    ci = bootstrap_ci(rows, n_boot=200).iloc[0]
    assert ci.before_lo <= ci.before_hi and ci.after_lo <= ci.after_hi


def test_report_builds(tmp_path):
    from voxguard.bench.dataset import expand_accents
    from voxguard.bench.report import build_report
    from voxguard.bench.runner import save_run
    cases = expand_accents(generate_cases(5), ["us", "scottish"])
    rows, summ = asyncio.run(run_benchmark(cases, MockAgent(), MockShadow()))
    save_run(rows, summ, str(tmp_path / "runs"))
    out = build_report(str(tmp_path / "runs"), str(tmp_path / "report"))
    for f in ["REPORT.md", "silent_error.png", "accents.png", "latency.png"]:
        assert (out / f).exists()
