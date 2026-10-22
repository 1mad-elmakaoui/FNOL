import io
import wave

import numpy as np
import pytest

import app.agent.loop as loop_module
from app.db.models import Call, CallStatus
from tests.fakes import ScriptedClient, response, tool_use


def wav_bytes(seconds=1.0, rate=16000):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def test_health_and_schemas(client):
    assert client.get("/health").json()["status"] == "ok"
    names = [s["name"] for s in client.get("/schemas").json()]
    assert names == ["fnol", "kyc"]
    assert client.get("/schemas/fnol").json()["$id"] == "fnol"
    assert client.get("/schemas/nope").status_code == 404


def test_upload_queues_call_and_serves_audio_with_ranges(client, db):
    data = wav_bytes()
    r = client.post("/calls", files={"file": ("my call.wav", data, "audio/wav")})
    assert r.status_code == 201
    call_id = r.json()["call_id"]
    assert r.json()["status"] == CallStatus.QUEUED
    detail = client.get(f"/calls/{call_id}").json()
    assert detail["status"] == "queued" and detail["has_audio"] and detail["segments"] == []
    assert any(c["id"] == call_id for c in client.get("/calls").json())

    full = client.get(f"/calls/{call_id}/audio")
    assert full.status_code == 200 and full.content == data
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-type"].startswith("audio/")

    part = client.get(f"/calls/{call_id}/audio", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206
    assert part.content == data[100:200]
    assert part.headers["content-range"] == f"bytes 100-199/{len(data)}"
    assert part.headers["content-length"] == "100"

    tail = client.get(f"/calls/{call_id}/audio", headers={"Range": "bytes=-50"})
    assert tail.status_code == 206 and tail.content == data[-50:]
    open_ended = client.get(f"/calls/{call_id}/audio", headers={"Range": f"bytes={len(data) - 10}-"})
    assert open_ended.content == data[-10:]
    bad = client.get(f"/calls/{call_id}/audio", headers={"Range": f"bytes={len(data) + 5}-"})
    assert bad.status_code == 416 and bad.headers["content-range"] == f"bytes */{len(data)}"

    # Extraction is refused while the call is still queued.
    assert client.post(f"/calls/{call_id}/extract", json={"schema": "fnol"}).status_code == 409

    # Right to erasure.
    assert client.delete(f"/calls/{call_id}").status_code == 204
    assert client.get(f"/calls/{call_id}").status_code == 404


def test_upload_rejects_non_audio(client):
    r = client.post("/calls", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_extract_without_api_key_returns_clear_503(client, seeded):
    r = client.post(f"/calls/{seeded['clean_call']}/extract", json={"schema": "fnol"})
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "configuration"
    assert "ANTHROPIC_API_KEY is not set" in body["detail"]
    assert "Traceback" not in r.text


def test_extract_unknown_schema(client, seeded):
    r = client.post(f"/calls/{seeded['clean_call']}/extract", json={"schema": "nope"})
    assert r.status_code == 404 and r.json()["error"] == "schema_not_found"


@pytest.fixture()
def fake_model(monkeypatch):
    def install(script):
        fake = ScriptedClient(script)
        monkeypatch.setattr(loop_module, "make_client", lambda settings=None: fake)
        return fake
    return install


def test_extract_then_review_and_correct(client, seeded, db, fake_model):
    call_id = seeded["clean_call"]
    detail = client.get(f"/calls/{call_id}").json()
    plate_seg = next(s for s in detail["segments"] if "GE-418-TR" in s["text"])
    fake_model([
        response(tool_use("lookup_policy", policy_number="FNX-204518")),
        response(tool_use(
            "submit_form",
            payload={"vehicle_plate": "GE-418-TR", "injuries_reported": False},
            evidence=[{"field_path": "vehicle_plate", "start_ms": plate_seg["start_ms"],
                       "end_ms": plate_seg["end_ms"], "confidence": 0.97}],
        )),
    ])
    r = client.post(f"/calls/{call_id}/extract", json={"schema": "fnol"})
    assert r.status_code == 200, r.text
    ext = r.json()
    assert ext["status"] == "completed" and ext["termination_reason"] == "submitted"
    plate = next(f for f in ext["fields"] if f["field_path"] == "vehicle_plate")
    assert plate["start_ms"] == plate_seg["start_ms"] and "GE-418-TR" in plate["quote"]
    assert plate["confidence"] == 0.97
    assert {m["field_path"] for m in ext["missing"]} >= {"policy_number", "callback_number"}
    assert len(ext["schema_fields"]) == 16
    assert client.get(f"/calls/{call_id}").json()["status"] == CallStatus.EXTRACTED

    eid = ext["id"]
    # Invalid correction is rejected with the validator's message.
    bad = client.patch(f"/extractions/{eid}/fields/vehicle_plate", json={"value": "not a plate"})
    assert bad.status_code == 422 and "license-plate" in str(bad.json())

    # Correct an agent value: original kept, corrected value normalized.
    ok = client.patch(f"/extractions/{eid}/fields/vehicle_plate", json={"value": "ge 418 tx"}).json()
    plate = next(f for f in ok["fields"] if f["field_path"] == "vehicle_plate")
    assert plate["value"] == "GE-418-TX" and plate["original_value"] == "GE-418-TR" and plate["corrected"]
    assert ok["payload"]["vehicle_plate"] == "GE-418-TX"
    # A second correction keeps the *agent's* original.
    ok = client.patch(f"/extractions/{eid}/fields/vehicle_plate", json={"value": "GE-418-TR"}).json()
    plate = next(f for f in ok["fields"] if f["field_path"] == "vehicle_plate")
    assert plate["original_value"] == "GE-418-TR" and len(ok["corrections"]) == 2

    # Fill a missing field: it is resolved and leaves the follow-up script.
    ok = client.patch(f"/extractions/{eid}/fields/callback_number", json={"value": "06 12 45 78 90"}).json()
    assert ok["payload"]["callback_number"] == "0612457890"
    assert next(m for m in ok["missing"] if m["field_path"] == "callback_number")["resolved"]
    assert "reach you on" not in ok["followup_script"]

    # Unknown field / extraction.
    assert client.patch(f"/extractions/{eid}/fields/colour", json={"value": "red"}).status_code == 404
    assert client.get("/extractions/9999").status_code == 404
    traced = client.get(f"/extractions/{eid}?trace=true").json()
    assert [t["tool"] for t in traced["trace"]] == ["lookup_policy", "submit_form"]


def test_nested_field_path_patch(client, seeded, fake_model):
    fake_model([response(tool_use("submit_form", payload={"address": {"city": "Lyon"}}, evidence=[]))])
    ext = client.post(f"/calls/{seeded['french_call']}/extract", json={"schema": "kyc"}).json()
    assert ext["payload"] == {"address": {"city": "Lyon"}}
    ok = client.patch(f"/extractions/{ext['id']}/fields/address.postal_code", json={"value": "69003"}).json()
    assert ok["payload"]["address"] == {"city": "Lyon", "postal_code": "69003"}


def test_extract_while_extracting_is_rejected(client, seeded, db):
    call = db.get(Call, seeded["clean_call"])
    call.status = CallStatus.EXTRACTING
    db.commit()
    assert client.post(f"/calls/{call.id}/extract", json={}).status_code == 409
