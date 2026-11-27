"""Three layers, tested on failures seen in real runs (Deepgram Flux, Nova-3, AssemblyAI)."""
from voxguard.core.types import ShadowTranscript, ToolCall, TurnContext
from voxguard.layers import apply_reply, attribute, heard_digits, plan_repair, recognize_entity
from voxguard.names import NameGuard
from voxguard.numbers import NumberGuard


def test_layer1_reads_spoken_and_fragmented_numbers():
    assert heard_digits("eight eight eight two seven two two zero twenty eight", 10) == "8882722028"
    assert heard_digits("My cell | is nine four six | four eight nine | three nine | double four.", 10) == "9464893944"
    assert recognize_entity("phone", "9464893944", "nine four six four eight nine three nine double four") == "correct"
    assert recognize_entity("phone", "9464893944", "nine four six four eight nine three nine four") == "wrong"
    assert recognize_entity("name", "Tadhg Brennan", "my name is Tadhg Brennan") == "correct"
    assert recognize_entity("name", "Fahad Rahman", "this is Fahad") == "partial"
    assert recognize_entity("phone", "9464893944", None) is None


def test_layer2_attribution():
    assert attribute("correct", "correct") == "ok"
    assert attribute("wrong", "correct") == "understanding"       # heard right, LLM wrote it wrong
    assert attribute("wrong", "wrong") == "recognition"
    assert attribute("wrong", "sound_alike") == "spelling"
    assert attribute("wrong", "wrong", cut_in=True) == "turn_taking"
    assert attribute("wrong", None) == "unknown"


def _ctx(heard, source="pipeline", shadow=None):
    return TurnContext(shadow=ShadowTranscript(shadow, 0.9) if shadow else None,
                       extra={"heard": heard, "heard_source": source})


def test_layer2_guard_redoes_the_llms_number_conversion():
    g = NumberGuard(fields={"phone": "phone"})
    # Flux heard it right; gpt-4o-mini wrote 11 digits
    v = g.check_tool_call(ToolCall("record_details", {"phone": "88827222028"}),
                          _ctx("Yeah. My number is eight eight eight two seven two two zero twenty eight."))[0]
    assert v.action.value == "correct" and v.value == "8882722028"
    # side-channel transcript alone (speech-to-speech model) is not enough to overwrite
    v = g.check_tool_call(ToolCall("record_details", {"phone": "88827222028"}),
                          _ctx("eight eight eight two seven two two zero twenty eight", "side_channel"))[0]
    assert v.action.value != "correct"
    # ...but it is when the second recognizer agrees
    v = g.check_tool_call(ToolCall("record_details", {"phone": "88827222028"}),
                          _ctx("eight eight eight two seven two two zero twenty eight", "side_channel",
                               "my number is 888 272 2028"))[0]
    assert v.action.value == "correct" and v.value == "8882722028"


def test_layer2_flags_a_phone_number_the_tool_call_left_out():
    g = NumberGuard(fields={"phone": "phone", "order_id": "order_id"})
    vs = g.check_tool_call(ToolCall("record_details", {"name": "Ana"}), _ctx(None, shadow="call me on 617 555 0188"))
    assert [(v.field, v.action.value) for v in vs] == [("phone", "confirm")]
    # the same digits already recorded as the phone are not a missing order id
    vs = g.check_tool_call(ToolCall("record_details", {"phone": "6175550188"}), _ctx(None, shadow="617 555 0188"))
    assert all(v.field == "phone" for v in vs)


def test_layer2_partial_name():
    g = NameGuard()
    ctx = TurnContext(shadow=ShadowTranscript("my name is Fahad Rahman", 0.9), roster=["Fahad Rahman", "Ewa Kowalczyk"])
    v = g.check_tool_call(ToolCall("record_details", {"name": "Fahad"}), ctx)[0]
    assert v.action.value == "correct" and v.value == "Fahad Rahman"
    ctx = TurnContext(shadow=ShadowTranscript("my name is Fahad Rahman", 0.9))
    v = g.check_tool_call(ToolCall("record_details", {"name": "Fahad"}), ctx)[0]
    assert v.action.value == "confirm"


def test_layer3_targeted_repair_asks_less_than_full():
    for model, evidence, truth in [("7192093799", "7192098799", "7192098799"),
                                   ("946489393944", "9464893944", "9464893944")]:
        t = plan_repair("phone", "phone", model, evidence, truth, "targeted")
        f = plan_repair("phone", "phone", model, evidence, truth, "full")
        assert t.units < f.units == 10
        assert apply_reply(t, model, [a.reply for a in t.asks], "phone") == truth
        assert apply_reply(f, model, [a.reply for a in f.asks], "phone") == truth
    # a false alarm costs a "yes", not a repeat
    t = plan_repair("phone", "phone", "7192098799", "7192093799", "7192098799", "targeted")
    assert t.asks[0].confirms and t.units == 0
    # a name: spell only the part that disagrees, or the part that is missing
    t = plan_repair("name", "name", "Tyg Brennan", "Teague Brennan", "Tadhg Brennan", "targeted")
    assert [a.token for a in t.asks] == [0]
    assert apply_reply(t, "Tyg Brennan", ["T, A, D, H, G."], "name") == "Tadhg Brennan"
    t = plan_repair("name", "name", "Fahad", "Fahad Rahman", "Fahad Rahman", "targeted")
    assert apply_reply(t, "Fahad", ["R A H M A N"], "name") == "Fahad Rahman"


def test_recover_end_to_end_offline(tmp_path):
    import asyncio

    import pandas as pd

    from voxguard.bench.recover import Recognizer, end_to_end, recover_rows
    rows = pd.DataFrame([
        {"case_id": "a", "field": "phone", "trial": 0, "model": "m", "accent": "us", "expected": "7192098799",
         "before_value": "7192093799", "action": "confirm", "before": "wrong", "after": "caught",
         "shadow_text": "seven one nine two zero nine eight seven nine nine", "agent_heard": None},
        {"case_id": "b", "field": "phone", "trial": 0, "model": "m", "accent": "us", "expected": "6175550188",
         "before_value": "6175550188", "action": "pass", "before": "correct", "after": "correct",
         "shadow_text": "", "agent_heard": None},
    ])
    rec = asyncio.run(recover_rows(rows, Recognizer(None, "text")))
    assert bool(rec.loc[0, "targeted_ok"]) and rec.loc[0, "targeted_units"] == 4
    e = end_to_end(rows, rec).iloc[0]
    assert e["without_voxguard"] == 0.5 and e["voxguard_targeted"] == 1.0
