"""Diarization fallback and word/speaker alignment (no ML models needed)."""
import numpy as np

from app.asr.align import assign_roles, build_segments
from app.asr.diarize import SAMPLE_RATE, Turn, diarize, energy_diarize, speech_spans
from app.asr.transcribe import Word


def tone(freq, seconds, amp):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


# Two "voices": a loud low one and a quiet high one, alternating with pauses.
PATTERN = ["A", "B", "A", "B", "B", "A"]
VOICE = {"A": (180, 0.5), "B": (1400, 0.08)}


def synthetic_call():
    pieces, spans, t = [], [], 0.0
    for who in PATTERN:
        freq, amp = VOICE[who]
        pieces.append(tone(freq, 1.5, amp))
        spans.append((int(t * 1000), int((t + 1.5) * 1000)))
        pieces.append(silence(0.7))
        t += 2.2
    return np.concatenate(pieces), spans


def test_speech_spans_merge_on_short_pauses():
    words = [(0, 300), (350, 700), (1500, 1800), (1850, 2000)]
    assert speech_spans(words, gap_ms=400) == [(0, 700), (1500, 2000)]


def test_energy_diarize_mono_separates_two_voices():
    audio, spans = synthetic_call()
    turns, method = energy_diarize(spans, audio)
    assert method == "energy-mono"
    labels = [t.speaker for t in turns]
    expected = ["SPEAKER_00" if w == "A" else "SPEAKER_01" for w in PATTERN]
    assert labels == expected


def test_energy_diarize_stereo_uses_channel_dominance():
    audio, spans = synthetic_call()
    left = np.concatenate([np.concatenate([tone(300, 1.5, 0.4) if w == "A" else silence(1.5), silence(0.7)]) for w in PATTERN])
    right = np.concatenate([np.concatenate([tone(300, 1.5, 0.4) if w == "B" else silence(1.5), silence(0.7)]) for w in PATTERN])
    turns, method = energy_diarize(spans, (left + right) / 2, (left, right))
    assert method == "energy-stereo"
    # Consecutive utterances on the same channel ("B", "B") merge into one turn.
    merged = [w for i, w in enumerate(PATTERN) if i == 0 or PATTERN[i - 1] != w]
    assert [t.speaker for t in turns] == ["SPEAKER_00" if w == "A" else "SPEAKER_01" for w in merged]


def test_stereo_splits_turns_with_short_gaps():
    """Turn-taking gaps shorter than the utterance gap must not merge two speakers."""
    left = np.concatenate([tone(300, 1.0, 0.4), silence(1.3)])  # agent: 0-1.0 s
    right = np.concatenate([silence(1.2), tone(300, 1.1, 0.4)])  # caller: 1.2-2.3 s (200 ms gap)
    words = [(50, 450), (500, 950), (1250, 1700), (1750, 2250)]
    spans = speech_spans(words)
    assert len(spans) == 1  # a single utterance span by pause alone
    turns, method = energy_diarize(spans, (left + right) / 2, (left, right), words)
    assert method == "energy-stereo"
    assert [(t.start_ms, t.end_ms, t.speaker) for t in turns] == [
        (50, 950, "SPEAKER_00"), (1250, 2250, "SPEAKER_01")]


def test_diarize_without_hf_token_logs_fallback(caplog):
    audio, spans = synthetic_call()
    turns, method = diarize(spans, audio)
    assert method.startswith("energy")
    assert "HF_TOKEN is not set" in caplog.text


def test_roles_first_voice_is_agent():
    assert assign_roles(["SPEAKER_01", "SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]) == {
        "SPEAKER_01": "agent", "SPEAKER_00": "caller", "SPEAKER_02": "speaker_02"}


def test_build_segments_groups_words_by_speaker_and_pause():
    words = [Word(" Hello", 0, 400), Word(" there.", 450, 800),
             Word(" My", 1000, 1200), Word(" plate", 1250, 1500), Word(" is", 1550, 1700),
             Word(" GE-418-TR.", 3500, 4200)]
    turns = [Turn(0, 850, "SPEAKER_00"), Turn(950, 4300, "SPEAKER_01")]
    segs = build_segments(words, turns)
    assert [(s.speaker, s.text) for s in segs] == [
        ("agent", "Hello there."), ("caller", "My plate is"), ("caller", "GE-418-TR.")]  # split on the 1.8 s pause
    assert segs[1].start_ms == 1000 and segs[1].end_ms == 1700
    assert segs[2].words_json()[0] == {"w": "GE-418-TR.", "s": 3500, "e": 4200, "p": 1.0}
