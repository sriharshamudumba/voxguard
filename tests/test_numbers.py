from voxguard.core.types import Action, ShadowTranscript, ToolCall, TurnContext
from voxguard.numbers import DigitChannelDecoder, NumberGuard, extract_digit_runs, luhn, us_phone, words_to_digits


def test_spoken_digits_basic():
    assert words_to_digits("six seven eight, nine three two, oh two double eight") == "6789320288"


def test_spoken_hundred_and_pairs():
    assert words_to_digits("eight hundred, five five five, twelve thirty four") == "8005551234"
    assert words_to_digits("twenty three") == "23"
    assert words_to_digits("triple seven") == "777"


def test_oh_only_counts_next_to_numbers():
    assert extract_digit_runs("oh hi there") == []
    assert words_to_digits("four oh four") == "404"


def test_validators():
    assert us_phone("6789320288")
    assert not us_phone("1789320288")
    assert not us_phone("678932028")
    assert luhn("4111111111111111")
    assert not luhn("4111111111111112")


def _ctx(text, conf=0.92, **kw):
    return TurnContext(shadow=ShadowTranscript(text, conf), **kw)


def test_padding_is_caught():
    g = NumberGuard()
    v = g.check_tool_call(ToolCall("x", {"phone": "678-932-0288"}),
                          _ctx("my number is six seven eight nine three two zero two eight"))[0]
    assert v.action == Action.CONFIRM
    assert "did not say" in v.reason


def test_agreement_passes():
    g = NumberGuard()
    v = g.check_tool_call(ToolCall("x", {"phone": "6789320288"}),
                          _ctx("six seven eight, nine three two, zero two eight eight"))[0]
    assert v.action == Action.PASS


def test_disagreement_confirms():
    g = NumberGuard()
    v = g.check_tool_call(ToolCall("x", {"phone": "6789320988"}),
                          _ctx("six seven eight, nine three two, zero two eight eight"))[0]
    assert v.action == Action.CONFIRM


def test_known_record_correction():
    g = NumberGuard(fields={"order_id": "order_id"})
    ctx = _ctx("order number four two nine one seven three", known_values={"order_id": ["429173", "118822"]})
    v = g.check_tool_call(ToolCall("x", {"order_id": "429178"}), ctx)[0]
    assert v.action == Action.CORRECT and v.value == "429173"


def test_channel_decoder_prefers_valid():
    dec = DigitChannelDecoder()
    ranked = dec.decode("1789320288", valid=us_phone)
    assert ranked and us_phone(ranked[0][0])
