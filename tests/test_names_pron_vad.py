import numpy as np

from voxguard.core.decision import TemperatureScaler, expected_calibration_error
from voxguard.core.types import Action, ShadowTranscript, ToolCall, TurnContext
from voxguard.names import NameGuard, NameResolver, phonetic_key
from voxguard.pronunciation import normalize_for_tts
from voxguard.vad import EntityAwareEndpointer, FixedEndpointer, classify_state


def test_phonetic_key_groups_soundalikes():
    assert phonetic_key("Catherine") == phonetic_key("Kathryn")
    assert phonetic_key("Philip") == phonetic_key("Filip")


def test_roster_correction():
    roster = ["Siobhan Murphy|Shivawn Murphy", "Catherine Lee", "Jon Smith"]
    g = NameGuard()
    ctx = TurnContext(shadow=ShadowTranscript("hi it's under Shivon Murphy", 0.9), roster=roster)
    v = g.check_tool_call(ToolCall("x", {"name": "Shivon Murphy"}), ctx)[0]
    assert v.action == Action.CORRECT and v.value == "Siobhan Murphy"


def test_unknown_disagreement_asks_to_spell():
    g = NameGuard()
    ctx = TurnContext(shadow=ShadowTranscript("this is Tadhg Brennan", 0.9), roster=["Jon Smith"])
    v = g.check_tool_call(ToolCall("x", {"name": "Tiger Brennan"}), ctx)[0]
    assert v.action == Action.CONFIRM


def test_unknown_agreement_passes():
    g = NameGuard()
    ctx = TurnContext(shadow=ShadowTranscript("this is Tadhg Brennan", 0.9), roster=["Jon Smith"])
    v = g.check_tool_call(ToolCall("x", {"name": "Tadhg Brennan"}), ctx)[0]
    assert v.action == Action.PASS


def test_resolver_posterior_sums_to_one():
    hyps = NameResolver().resolve(["Sharon Murphy"], ["Siobhan Murphy", "Jon Smith"])
    assert abs(sum(h.posterior for h in hyps) - 1) < 1e-9


def test_normalizer():
    assert normalize_for_tts("Call 678-932-0288") == "Call six seven eight, nine three two, zero two eight eight"
    assert normalize_for_tts("That is $12.50") == "That is twelve dollars and fifty cents"
    assert normalize_for_tts("See you at 7:30 PM") == "See you at seven thirty p m"
    assert "dot com" in normalize_for_tts("email jo.lee@example.com")
    assert normalize_for_tts("Hi Siobhan", {"Siobhan": "Shivawn"}) == "Hi Shivawn"


def test_endpoint_states_and_thresholds():
    assert classify_state("my number is six seven eight") == "mid_number"
    assert classify_state("my number is six seven eight nine three two zero two eight eight") == "complete"
    assert classify_state("it's under") == "filler"
    ep = EntityAwareEndpointer()
    assert ep.threshold_ms("mid_number") > FixedEndpointer().ms > ep.threshold_ms("complete")


def test_temperature_scaling_reduces_ece():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.5, 0.9, 4000)
    y = rng.random(4000) < true_p
    overconf = 1 / (1 + np.exp(-3 * np.log(true_p / (1 - true_p))))
    ts = TemperatureScaler().fit(overconf, y)
    assert expected_calibration_error(ts.transform(overconf), y) < expected_calibration_error(overconf, y)
