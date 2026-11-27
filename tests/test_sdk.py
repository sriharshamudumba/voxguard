"""The live-agent SDK and HTTP service."""
import json
import threading
import urllib.request

from voxguard.sdk import Guard, summarize_log


def test_sdk_fixes_the_llms_number_and_name(tmp_path):
    g = Guard(roster=["Fahad Rahman", "Ewa Kowalczyk"], log_path=str(tmp_path / "log.jsonl"))
    call = g.new_call("c1")
    call.add_transcript("Yeah. My number is eight eight eight two seven two two zero twenty eight.")
    d = call.check("record_details", {"phone": "88827222028"})
    assert d.action == "correct" and d.args["phone"] == "8882722028"
    call.end_turn()
    call.add_transcript("it's under Fahad Rahman")
    d = call.check("record_details", {"name": "Fahad"})
    assert d.action == "correct" and d.args["name"] == "Fahad Rahman"


def test_sdk_asks_a_targeted_question_and_logs_the_followup(tmp_path):
    log = tmp_path / "log.jsonl"
    g = Guard(log_path=str(log))
    call = g.new_call("c2")
    call.add_transcript("call me on 617 555 0188")
    d = call.check("record_details", {"name": "Ana"})         # the LLM left the phone number out
    assert d.action == "confirm" and "VALIDATION_FAILED" in d.say and d.question
    call.end_turn()
    call.add_transcript("six one seven five five five zero one eight eight")
    d = call.check("record_details", {"name": "Ana", "phone": "6175550188"})
    assert d.action in ("pass", "correct")
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert rows[0]["args_in"] == {"name": "A**"} and "*" in rows[0]["heard"]   # redacted by default
    out = summarize_log(str(log))
    assert "1 calls, 2 tool calls checked" in out


def test_server_check_endpoint(tmp_path):
    from http.server import ThreadingHTTPServer

    from voxguard.server import make_handler
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Guard(log_path=str(tmp_path / "l.jsonl")), "t0k"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/check"
    body = json.dumps({"call_id": "x", "tool": "record_details", "args": {"phone": "88827222028"},
                       "transcript": "eight eight eight two seven two two zero twenty eight"}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json", "Authorization": "Bearer t0k"})
    d = json.loads(urllib.request.urlopen(req).read())
    assert d["action"] == "correct" and d["args"]["phone"] == "8882722028"
    bad = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(bad)
        raise AssertionError("expected 401")
    except urllib.error.HTTPError as e:
        assert e.code == 401
    srv.shutdown()


def test_sdk_asks_once_per_field_and_new_name_policy():
    g = Guard(roster=["Fahad Rahman"])
    call = g.new_call()
    assert call.check("record_details", {"name": "Ana"}).action == "confirm"     # not on the list: spell it
    call.end_turn()
    call.add_transcript("A, N, A")
    assert call.check("record_details", {"name": "Ana"}).action == "pass"        # answered: no second ask
    g2 = Guard(roster=["Fahad Rahman"], ask_unlisted_names=False)
    assert g2.new_call().check("record_details", {"name": "Ana"}).action == "pass"
