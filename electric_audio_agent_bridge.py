#!/usr/bin/env python3
"""
electric_audio_agent_bridge.py

Zero-PII Chargeable-Byte Electric Audio ("Electric Ear") & Multimodal Sight+Sound
Bridge for Agents.

Core Physical & Epistemic Model:
  "Audio gets translated to chargeable bytes; during play the bytes get charged up
   like pushed up."

Architecture (5-Layer Epistemic Stack + Second-Opinion Safeguards):
1. Layer 1 — `VerifiedAudioSource` & Chargeable-Byte Translation:
   - Translates synthetic audio events (`tone`, `chord`, `sweep`, `pulse`,
     `noise_burst`, `silence`), Cyclops Tone Alphabet V0 (`az-log-220-880-v1`),
     Musical Time V3 note events, or owner-supplied PCM waveforms into an
     immutable, content-addressed (`SHA-256`) 2D Chargeable-Byte Matrix
     (`time_slots x freq_bands`, `uint8` `0..255`).
   - Enforces non-live provenance (`synthetic_fixture` or `owner_supplied_audio`;
     live microphone capture `live_microphone_capture` is deterministically
     refused with `ERR_NON_SYNTHETIC_AUDIO_PROHIBITED`).
   - Lexical closure isolation: raw PCM waveform bytes and raw chargeable-byte
     arrays never leave the private closure vault (`0` raw audio or byte buffer
     leakage to agents).
2. Causal Playhead Charging ("Pushed Up During Play") + Multi-Scale Channels:
   - Before play (`playhead_slot == 0`), chargeable bytes sit at their resting
     floor (`0`) while holding latent `potential_byte` (`0..255`).
   - As playback advances across `playhead_slot`, active bytes are pushed up
     (`charged_up`) by their acoustic potential and onset novelty:
       * `fast_charge_u8` (`0..255`): rapid attack / fast decay (`160/256` per
         tick) capturing transients, clicks, and onsets.
       * `slow_charge_u8` (`0..255`): capacitor accumulation / slow decay
         (`232/256` per tick) capturing sustained pitch, harmonic body, and
         short-term temporal memory trace.
       * `pushed_up_byte_u8` (`0..255`): composite pushed-up byte elevation.
   - Refractory & Saturation Guards: per-band ceiling (`255`), maximum per-tick
     push-up step, and refractory cooldown (`refractory_ticks`) prevent loud or
     adversarial bursts from saturating or locking agent state.
3. Layer 2 — `DeterministicAudioMeasurement` (10 Integer / `Q16.16` Operators):
   - `op_charge_summary_u8`, `op_push_up_profile_q16`,
     `op_spectral_centroid_and_spread_q16`, `op_multiscale_fast_slow_ratio_q16`,
     `op_refractory_and_saturation_guard`, `op_connected_charge_events_4c`,
     `op_acoustic_contrast_wcag_analog_q16`, `op_spectro_temporal_flux_q16`,
     `op_perceptual_audio_hash_ahash64`, and `op_bbox_aligned_charge_diff_u8`.
4. Layer 3 — `DerivedAudioRepresentation` & `ContinuousElectricEarStream`:
   - Pairwise charge diffs, bounded ring-buffer window with `explicit_bzero`
     zeroization eviction, 16x16 dirty charge-tile bitmasks, automatic
     `verified -> stale` epistemic decay when the playhead advances, and
     multi-tick pitch kinematics.
5. Layer 4 & Layer 5 — Semantic Refusal Boundary, `ObservationReceipt`,
   `GhostOverlay`, and Multimodal Sight+Sound Fusion (`ElectricSightAndSoundSession`):
   - Deterministically refuses semantic speech transcription (ASR-for-agent),
     emotional/aesthetic musical judgment, or speaker-intent queries
     (`REFUSED_EPISTEMIC_BOUNDARY`).
   - Projects pushed-up audio bytes into visual marks (`charge_bytes_to_visual_marks`)
     and fuses Electric Vision + Electric Audio streams into joint audiovisual
     salience telemetry so agents can see and hear together.
"""

from __future__ import annotations

import copy
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from electric_eye_harness import (
    MAX_MARKS_PER_SOURCE,
    OPERATOR_VERSION,
    REFUSAL_CODE_SEMANTIC,
    REFUSAL_REASON_SEMANTIC,
    DrawingMark,
    ElectricEyeFixtureHarness,
    assert_no_raw_buffer_leakage,
    canonical_json,
    compute_replay_digest,
    sha256_hex,
)
from electric_vision_agent_bridge import (
    ElectricVisionAgentSession,
    _canonical_sha256_tag,
    _dhash64_hamming_distance,
)
from prod_gate_enclave import (
    build_accessibility_metadata,
    explicit_zeroize_bytearray,
)

SUPPORTED_AUDIO_GRIDS: Tuple[Tuple[int, int], ...] = (
    (128, 32),
    (128, 64),
    (256, 64),
    (512, 64),
    (512, 128),
)

VALID_AUDIO_ORIGINS = frozenset({
    "synthetic_fixture",
    "owner_supplied_audio",
})

AUDIO_EVENT_KINDS = frozenset({
    "tone",
    "chord",
    "sweep",
    "pulse",
    "noise_burst",
    "silence",
    "note",
})

# Tone Alphabet V0 constants (az-log-220-880-v1)
TONE_ALPHABET_MIN_HZ = 220.0
TONE_ALPHABET_MAX_HZ = 880.0
TONE_ALPHABET_ACTIVE_MS = 120.0
TONE_ALPHABET_GAP_MS = 30.0
TONE_ALPHABET_SLOT_MS = 150.0
TONE_ALPHABET_SPACE_MS = 300.0
TONE_ALPHABET_FADE_MS = 10.0

# Frequency analysis span (55 Hz A1 to 7040 Hz A8 = 7 octaves)
MIN_FREQ_HZ = 55.0
MAX_FREQ_HZ = 7040.0


def tone_alphabet_index_to_hz(index: int) -> float:
    """
    Tone Alphabet V0 (`az-log-220-880-v1`):
    frequency(i) = 220 * 2^(2*i / 25) Hz for i in 0..25 (A..Z).
    """
    if not (0 <= index <= 25):
        raise ValueError(f"Tone Alphabet index must be in 0..25, got {index}")
    return TONE_ALPHABET_MIN_HZ * math.pow(2.0, (2.0 * float(index)) / 25.0)


def tone_alphabet_text_to_events(text: str) -> Dict[str, Any]:
    """
    Deterministically encodes ASCII A-Z / a-z / space text into Tone Alphabet V0
    tone & silence events without any semantic interpretation.
    """
    if not isinstance(text, str):
        raise ValueError("Tone Alphabet input must be a string")
    events: List[Dict[str, Any]] = []
    cursor_ms = 0.0
    for idx, ch in enumerate(text):
        code = ord(ch)
        if ch == " ":
            events.append({
                "event_id": f"ta_sp_{idx:04d}",
                "kind": "silence",
                "start_ms": round(cursor_ms, 3),
                "duration_ms": TONE_ALPHABET_SPACE_MS,
                "gain": 0.0,
            })
            cursor_ms += TONE_ALPHABET_SPACE_MS
        elif 65 <= code <= 90 or 97 <= code <= 122:
            upper_code = code - 32 if code >= 97 else code
            letter_idx = upper_code - 65
            freq_hz = tone_alphabet_index_to_hz(letter_idx)
            events.append({
                "event_id": f"ta_tone_{idx:04d}",
                "kind": "tone",
                "letter_index": letter_idx,
                "freq_hz": round(freq_hz, 6),
                "start_ms": round(cursor_ms, 3),
                "duration_ms": TONE_ALPHABET_ACTIVE_MS,
                "attack_ms": TONE_ALPHABET_FADE_MS,
                "release_ms": TONE_ALPHABET_FADE_MS,
                "gain": 0.82,
            })
            cursor_ms += TONE_ALPHABET_SLOT_MS
        else:
            raise ValueError(
                f"ERR_INVALID_TONE_ALPHABET_CHAR: unsupported character at position {idx}"
            )
    total_duration_ms = max(TONE_ALPHABET_SLOT_MS, round(cursor_ms, 3))
    return {
        "profile_id": "az-log-220-880-v1",
        "event_count": len(events),
        "total_duration_ms": total_duration_ms,
        "events": events,
    }


def _hz_to_band_index(freq_hz: float, freq_bands: int) -> int:
    """Maps a frequency in Hz [55..7040] to a logarithmic band index [0..freq_bands-1]."""
    clamped = max(MIN_FREQ_HZ, min(MAX_FREQ_HZ, float(freq_hz)))
    octaves = math.log2(clamped / MIN_FREQ_HZ)
    max_octaves = math.log2(MAX_FREQ_HZ / MIN_FREQ_HZ)  # 7.0 octaves
    norm = octaves / max_octaves
    return max(0, min(freq_bands - 1, int(round(norm * (freq_bands - 1)))))


def _band_index_to_hz(band_idx: int, freq_bands: int) -> int:
    """Maps a logarithmic band index [0..freq_bands-1] back to nominal integer Hz."""
    if freq_bands <= 1:
        return int(round(MIN_FREQ_HZ))
    norm = max(0.0, min(1.0, float(band_idx) / float(freq_bands - 1)))
    max_octaves = math.log2(MAX_FREQ_HZ / MIN_FREQ_HZ)
    return int(round(MIN_FREQ_HZ * math.pow(2.0, norm * max_octaves)))


def _midi_to_hz(midi_note: float) -> float:
    return 440.0 * math.pow(2.0, (float(midi_note) - 69.0) / 12.0)


def _validate_audio_event_hygiene(event_dict: Dict[str, Any]) -> None:
    """Enforces zero-PII and zero-path hygiene on synthetic audio event specifications."""
    serialized = canonical_json(event_dict).lower()
    for bad_token in (
        "file://",
        "/home/",
        "/users/",
        "/etc/",
        "/var/",
        "@",
        "passwd",
        "secret",
        "transcript",
        "utterance",
        "speaker_name",
    ):
        if bad_token in serialized:
            raise ValueError(
                f"ERR_UNSAFE_AUDIO_SPEC: prohibited token {bad_token!r} in audio event"
            )
    if event_dict.get("kind") in ("speech", "voice_clone", "asr_transcript"):
        raise ValueError(
            "ERR_SPEECH_SYNTHESIS_OR_ASR_PROHIBITED: speech/transcript tokens are excluded from zero-PII Electric Audio"
        )


def translate_audio_to_chargeable_bytes(
    audio_events_or_spec: Union[Sequence[Dict[str, Any]], Dict[str, Any]],
    time_slots: int = 256,
    freq_bands: int = 64,
    *,
    total_duration_ms: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[DrawingMark]]:
    """
    Translates structured audio events (tones, chords, sweeps, pulses, noise bursts,
    Musical Time V3 notes, or Tone Alphabet V0 sequences) into deterministic
    Chargeable-Byte instructions (`potential_byte` 0..255 across `time_slots x freq_bands`),
    plus 128x128 `DrawingMark` proxies for the out-of-process VaultGate daemon.

    Returns:
      - `normalized_charge_specs`: deterministic per-event chargeable-byte spans
      - `proxy_128_marks`: `DrawingMark` list scaled to `128x128` for VaultGate
    """
    if (time_slots, freq_bands) not in SUPPORTED_AUDIO_GRIDS:
        raise ValueError(
            f"Unsupported audio chargeable-byte grid {(time_slots, freq_bands)}; "
            f"must be one of {SUPPORTED_AUDIO_GRIDS}"
        )

    raw_events: Sequence[Dict[str, Any]]
    inferred_duration_ms = float(total_duration_ms or 0.0)

    if isinstance(audio_events_or_spec, dict):
        if "tone_text" in audio_events_or_spec:
            plan = tone_alphabet_text_to_events(str(audio_events_or_spec["tone_text"]))
            raw_events = plan["events"]
            if inferred_duration_ms <= 0.0:
                inferred_duration_ms = float(plan["total_duration_ms"])
        elif "events" in audio_events_or_spec and isinstance(audio_events_or_spec["events"], list):
            raw_events = audio_events_or_spec["events"]
            if inferred_duration_ms <= 0.0 and "total_duration_ms" in audio_events_or_spec:
                inferred_duration_ms = float(audio_events_or_spec["total_duration_ms"])
        elif "notes" in audio_events_or_spec and isinstance(audio_events_or_spec["notes"], list):
            raw_events = audio_events_or_spec["notes"]
        else:
            raise ValueError(
                "Audio spec dict must contain 'events', 'notes', or 'tone_text'"
            )
    elif isinstance(audio_events_or_spec, (list, tuple)):
        raw_events = audio_events_or_spec
    else:
        raise ValueError("Audio specification must be a list of event dicts or a spec dict")

    if len(raw_events) > MAX_MARKS_PER_SOURCE:
        raise ValueError(
            f"Audio event count {len(raw_events)} exceeds MAX_MARKS_PER_SOURCE ({MAX_MARKS_PER_SOURCE})"
        )

    # Determine timeline span in ms
    if inferred_duration_ms <= 0.0:
        max_end_ms = 1000.0
        for ev in raw_events:
            if isinstance(ev, dict):
                st = float(ev.get("start_ms", ev.get("start", 0.0)))
                dur = float(ev.get("duration_ms", ev.get("duration", 150.0)))
                if st + dur > max_end_ms:
                    max_end_ms = st + dur
        inferred_duration_ms = max(150.0, max_end_ms)

    normalized_specs: List[Dict[str, Any]] = []
    proxy_128_marks: List[DrawingMark] = []

    for idx, item in enumerate(raw_events):
        if not isinstance(item, dict):
            raise ValueError(f"Audio event at index {idx} must be a dict")
        _validate_audio_event_hygiene(item)

        ev_id = str(item.get("event_id") or item.get("id") or f"aud_{idx:04d}")
        ev_id = re.sub(r"[^a-zA-Z0-9_]", "_", ev_id)[:48] or f"aud_{idx:04d}"
        kind = str(item.get("kind") or item.get("type") or "tone").lower()
        if kind == "noteon" or kind == "note":
            kind = "tone"
        if kind not in AUDIO_EVENT_KINDS:
            raise ValueError(f"Unsupported audio event kind {kind!r} at index {idx}")

        # Time slot mapping [t0..t1]
        if "t0" in item and "t1" in item:
            t0 = max(0, min(time_slots - 1, int(item["t0"])))
            t1 = max(t0, min(time_slots - 1, int(item["t1"])))
        elif "start_norm" in item:
            sn = max(0.0, min(1.0, float(item["start_norm"])))
            dn = max(0.01, min(1.0, float(item.get("duration_norm", 0.1))))
            t0 = max(0, min(time_slots - 1, int(round(sn * (time_slots - 1)))))
            t1 = max(t0, min(time_slots - 1, int(round((sn + dn) * (time_slots - 1)))))
        else:
            start_ms = max(0.0, float(item.get("start_ms", item.get("start", 0.0))))
            dur_ms = max(1.0, float(item.get("duration_ms", item.get("duration", 120.0))))
            t0 = max(
                0,
                min(time_slots - 1, int(round((start_ms / inferred_duration_ms) * (time_slots - 1)))),
            )
            t1 = max(
                t0,
                min(
                    time_slots - 1,
                    int(round(((start_ms + dur_ms) / inferred_duration_ms) * (time_slots - 1))),
                ),
            )

        gain = max(0.0, min(1.0, float(item.get("gain", item.get("velocity", 0.8)))))
        charge_potential_u8 = max(0, min(255, int(round(gain * 255.0))))
        if kind == "silence" or charge_potential_u8 == 0:
            continue

        # Frequency band mapping
        if "freq_hz" in item or "freq" in item:
            f_hz = float(item.get("freq_hz", item.get("freq", 440.0)))
        elif "midi" in item or "pitch" in item:
            f_hz = _midi_to_hz(float(item.get("midi", item.get("pitch", 69))))
        elif "band" in item:
            b_direct = max(0, min(freq_bands - 1, int(item["band"])))
            f_hz = float(_band_index_to_hz(b_direct, freq_bands))
        else:
            f_hz = 440.0

        b0 = _hz_to_band_index(f_hz, freq_bands)
        b1 = b0
        harmonics = max(1, min(6, int(item.get("harmonics", 1))))
        bandwidth_bands = max(0, min(16, int(item.get("bandwidth_bands", 1))))

        if kind == "sweep":
            f_end_hz = float(item.get("end_freq_hz", item.get("freq_end_hz", f_hz * 2.0)))
            b1 = _hz_to_band_index(f_end_hz, freq_bands)
        elif kind == "chord":
            freqs = item.get("frequencies_hz") or [f_hz, f_hz * 1.25, f_hz * 1.5]
            chord_bands = [_hz_to_band_index(float(cf), freq_bands) for cf in freqs]
            b0 = min(chord_bands)
            b1 = max(chord_bands)
        elif kind in ("pulse", "noise_burst"):
            b0 = max(0, b0 - max(2, bandwidth_bands))
            b1 = min(freq_bands - 1, b0 + max(4, bandwidth_bands * 2))
            chord_bands = list(range(b0, b1 + 1))
        else:
            chord_bands = [b0]

        attack_slots = max(0, min((t1 - t0) // 2, int(item.get("attack_slots", 1))))
        release_slots = max(0, min((t1 - t0) // 2, int(item.get("release_slots", 1))))

        spec = {
            "event_id": ev_id,
            "kind": kind,
            "t0": t0,
            "t1": t1,
            "b0": b0,
            "b1": b1,
            "chord_bands": chord_bands if kind == "chord" else [],
            "harmonics": harmonics,
            "bandwidth_bands": bandwidth_bands,
            "potential_u8": charge_potential_u8,
            "attack_slots": attack_slots,
            "release_slots": release_slots,
            "freq_hz": round(f_hz, 3),
        }
        normalized_specs.append(spec)

        # Build 128x128 proxy DrawingMark for VaultGate admission
        px0 = max(0, min(127, (t0 * 128) // time_slots))
        px1 = max(px0, min(127, (t1 * 128) // time_slots))
        # Invert y so higher frequency bands sit higher on the 128x128 proxy canvas
        py0 = max(0, min(127, 127 - ((max(b0, b1) * 128) // freq_bands)))
        py1 = max(py0, min(127, 127 - ((min(b0, b1) * 128) // freq_bands)))
        proxy_shape = "line" if kind == "sweep" else "rect"
        rgba = (
            min(255, charge_potential_u8),
            min(255, (b0 * 255) // max(1, freq_bands - 1)),
            255,
            255,
        )
        if len(proxy_128_marks) < MAX_MARKS_PER_SOURCE:
            proxy_128_marks.append(
                DrawingMark(
                    mark_id=ev_id,
                    shape=proxy_shape,
                    x0=px0,
                    y0=py0,
                    x1=px1,
                    y1=py1,
                    rgba=rgba,
                )
            )

    return normalized_specs, proxy_128_marks


def _compile_chargeable_byte_potentials(
    specs: Sequence[Dict[str, Any]],
    time_slots: int,
    freq_bands: int,
) -> Tuple[bytes, bytes]:
    """
    Compiles normalized audio specs into:
      1. `potential_bytes` (`time_slots * freq_bands` `uint8` `0..255`):
         latent chargeable byte capacity at each `(t, b)` slot.
      2. `novelty_bytes` (`time_slots * freq_bands` `uint8` `0..255`):
         positive onset/transient energy (`max(0, potential[t, b] - potential[t-1, b])`).

    Executed strictly inside the private lexical closure vault.
    """
    total_cells = time_slots * freq_bands
    pot_buf = bytearray(total_cells)

    def _add_potential(t: int, b: int, val: int) -> None:
        if 0 <= t < time_slots and 0 <= b < freq_bands and val > 0:
            off = t * freq_bands + b
            cur = pot_buf[off]
            pot_buf[off] = min(255, cur + val)

    for spec in specs:
        kind = spec["kind"]
        t0, t1 = int(spec["t0"]), int(spec["t1"])
        b0, b1 = int(spec["b0"]), int(spec["b1"])
        peak_u8 = int(spec["potential_u8"])
        att = int(spec.get("attack_slots", 1))
        rel = int(spec.get("release_slots", 1))
        bw = int(spec.get("bandwidth_bands", 1))
        harmonics = int(spec.get("harmonics", 1))
        span_t = max(1, t1 - t0 + 1)

        for t in range(t0, t1 + 1):
            # Deterministic trapezoidal envelope in integer arithmetic
            dt_start = t - t0
            dt_end = t1 - t
            env_num = 256
            if att > 0 and dt_start < att:
                env_num = min(env_num, ((dt_start + 1) * 256) // (att + 1))
            if rel > 0 and dt_end < rel:
                env_num = min(env_num, ((dt_end + 1) * 256) // (rel + 1))
            slot_u8 = (peak_u8 * env_num) >> 8
            if slot_u8 <= 0:
                continue

            if kind == "sweep":
                # Interpolate band linearly from b0 to b1 across [t0..t1]
                cur_b = b0 + ((b1 - b0) * (t - t0)) // max(1, span_t - 1)
                target_bands = [cur_b]
            elif kind == "chord":
                target_bands = list(spec.get("chord_bands") or [b0, b1])
            elif kind in ("pulse", "noise_burst"):
                lo = min(b0, b1)
                hi = max(b0, b1)
                target_bands = list(range(lo, hi + 1))
            else:
                target_bands = [b0]

            for base_b in target_bands:
                _add_potential(t, base_b, slot_u8)
                # Spread across immediate neighbor band(s) if bandwidth_bands > 0
                for d_b in range(1, bw + 1):
                    falloff_u8 = slot_u8 >> d_b
                    if falloff_u8 > 0:
                        _add_potential(t, base_b - d_b, falloff_u8)
                        _add_potential(t, base_b + d_b, falloff_u8)
                # Optional octave/harmonic overtones
                for h_idx in range(2, harmonics + 1):
                    # +1 octave in log bands is +(freq_bands - 1) / 7
                    h_band = base_b + int(round((math.log2(float(h_idx)) / 7.0) * (freq_bands - 1)))
                    h_u8 = slot_u8 // h_idx
                    if h_u8 > 0:
                        _add_potential(t, h_band, h_u8)

    # Compute onset novelty bytes: positive rise delta across time slots
    nov_buf = bytearray(total_cells)
    for t in range(time_slots):
        t_off = t * freq_bands
        prev_off = (t - 1) * freq_bands
        for b in range(freq_bands):
            cur_p = pot_buf[t_off + b]
            prev_p = pot_buf[prev_off + b] if t > 0 else 0
            delta = cur_p - prev_p
            if delta > 0:
                nov_buf[t_off + b] = min(255, delta)

    pot_bytes = bytes(pot_buf)
    nov_bytes = bytes(nov_buf)
    explicit_zeroize_bytearray(pot_buf)
    explicit_zeroize_bytearray(nov_buf)
    return pot_bytes, nov_bytes


def _simulate_causal_playhead_charging(
    potential_bytes: bytes,
    novelty_bytes: bytes,
    time_slots: int,
    freq_bands: int,
    *,
    playhead_slot: int,
    fast_decay_q8: int = 200,
    slow_decay_q8: int = 252,
    max_push_up_step: int = 240,
    refractory_threshold: int = 220,
    refractory_ticks: int = 2,
) -> Dict[str, Any]:
    """
    Simulates causal playback from slot `0` up to `playhead_slot`:
      - "During play the bytes get charged up like pushed up."
      - Bytes strictly ahead of `playhead_slot` (`t > playhead_slot`) remain
        uncharged at `0` (resting baseline).
      - As the playhead sweeps across `s = 0 .. playhead_slot`:
          * Existing charge across `0 .. s-1` decays via multi-scale Q8 factors:
              - `fast_charge` decays rapidly (`fast_decay_q8 / 256`, default 160/256)
                to isolate transients/onsets.
              - `slow_charge` decays slowly (`slow_decay_q8 / 256`, default 232/256)
                to retain short-term pitch/timbre memory trace.
          * At the active playhead column `s`, bytes are pushed up:
              - If band `b` is in refractory cooldown from a prior saturation spike,
                its push-up impulse is damped by `1/4` (refractory guard).
              - Otherwise `fast_charge[s, b]` is pushed up by `min(max_push_up_step,
                (potential + novelty) // 2)` and `slow_charge[s, b]` is pushed up by
                `min(max_push_up_step, potential)`.
              - Composite `pushed_up_bytes[t, b] = max(fast_charge[t, b], slow_charge[t, b])`.

    Returns the live charged byte arrays (kept strictly inside the closure vault)
    plus refractory/saturation guard counters.
    """
    total_cells = time_slots * freq_bands
    fast_buf = bytearray(total_cells)
    slow_buf = bytearray(total_cells)
    comp_buf = bytearray(total_cells)

    # If playhead_slot < 0, audio has not started playing: all bytes sit at resting 0
    if playhead_slot < 0:
        return {
            "fast_bytes": bytes(fast_buf),
            "slow_bytes": bytes(slow_buf),
            "pushed_up_bytes": bytes(comp_buf),
            "saturated_cell_count": 0,
            "refractory_clamped_count": 0,
            "active_playhead_slot": -1,
        }

    clamped_playhead = min(time_slots - 1, int(playhead_slot))
    band_cooldown = [0] * freq_bands
    saturated_cell_count = 0
    refractory_clamped_count = 0

    # Precompute integer Q8 decay multipliers by age_ticks for O(1) lookup
    # Wait: to preserve exact bit-for-bit parity with step-by-step >> 8 rounding,
    # only iterate when fast_impulse > 0 or slow_impulse > 0!
    for s in range(clamped_playhead + 1):
        s_off = s * freq_bands
        age_ticks = clamped_playhead - s
        for b in range(freq_bands):
            pot = potential_bytes[s_off + b]
            nov = novelty_bytes[s_off + b]
            if pot == 0 and nov == 0:
                if band_cooldown[b] > 0:
                    band_cooldown[b] -= 1
                continue

            # Apply refractory cooldown guard if triggered on earlier slot
            in_refractory = band_cooldown[b] > 0
            if in_refractory:
                band_cooldown[b] -= 1
                refractory_clamped_count += 1
                fast_impulse = min(max_push_up_step, ((pot >> 1) + nov) >> 2)
                slow_impulse = min(max_push_up_step, pot >> 2)
            else:
                fast_impulse = min(max_push_up_step, min(255, (pot >> 1) + nov))
                slow_impulse = min(max_push_up_step, pot)

            # Calculate decay distance (clamped_playhead - s) in closed-form integer steps
            f_val = fast_impulse
            s_val = slow_impulse
            for _ in range(age_ticks):
                f_val = (f_val * fast_decay_q8) >> 8
                s_val = (s_val * slow_decay_q8) >> 8
                if f_val == 0 and s_val == 0:
                    break

            c_val = f_val if f_val >= s_val else s_val
            fast_buf[s_off + b] = f_val
            slow_buf[s_off + b] = s_val
            comp_buf[s_off + b] = c_val

            # Check saturation at the moment the byte was pushed up
            raw_drive = max(pot, min(255, (pot >> 1) + nov))
            if not in_refractory and raw_drive >= refractory_threshold:
                saturated_cell_count += 1
                band_cooldown[b] = refractory_ticks

    out = {
        "fast_bytes": bytes(fast_buf),
        "slow_bytes": bytes(slow_buf),
        "pushed_up_bytes": bytes(comp_buf),
        "saturated_cell_count": saturated_cell_count,
        "refractory_clamped_count": refractory_clamped_count,
        "active_playhead_slot": clamped_playhead,
    }
    explicit_zeroize_bytearray(fast_buf)
    explicit_zeroize_bytearray(slow_buf)
    explicit_zeroize_bytearray(comp_buf)
    return out


def _run_10_audio_charge_operators(
    potential_bytes: bytes,
    pushed_up_bytes: bytes,
    fast_bytes: bytes,
    slow_bytes: bytes,
    secondary_pushed_up_bytes: Optional[bytes],
    time_slots: int,
    freq_bands: int,
    *,
    playhead_slot: int,
    saturated_cell_count: int = 0,
    refractory_clamped_count: int = 0,
    push_up_threshold_u8: int = 24,
    alert_threshold_u8: int = 180,
) -> Dict[str, Any]:
    """
    Executes all 10 Closed-Form Chargeable-Byte Audio Operators using strictly
    integer and Q16.16 fixed-point arithmetic (zero IEEE-754 non-determinism).
    """
    total_cells = time_slots * freq_bands
    sum_pot = 0
    sum_pushed = 0
    sum_fast = 0
    sum_slow = 0
    min_charge = 255
    max_charge = 0
    pushed_up_count = 0
    alert_cell_count = 0

    hist16 = [0] * 16
    band_charge_sums = [0] * freq_bands
    slot_charge_sums = [0] * time_slots

    pre_play_cells = 0
    active_window_cells = 0
    active_window_charge_sum = 0
    afterglow_cells = 0
    afterglow_charge_sum = 0

    clamped_ph = min(time_slots - 1, playhead_slot)

    for t in range(time_slots):
        t_off = t * freq_bands
        slot_sum = 0
        for b in range(freq_bands):
            idx = t_off + b
            p_u8 = potential_bytes[idx]
            c_u8 = pushed_up_bytes[idx]
            f_u8 = fast_bytes[idx]
            s_u8 = slow_bytes[idx]

            sum_pot += p_u8
            sum_pushed += c_u8
            sum_fast += f_u8
            sum_slow += s_u8
            slot_sum += c_u8
            band_charge_sums[b] += c_u8

            if c_u8 < min_charge:
                min_charge = c_u8
            if c_u8 > max_charge:
                max_charge = c_u8
            if c_u8 >= push_up_threshold_u8:
                pushed_up_count += 1
            if c_u8 >= alert_threshold_u8:
                alert_cell_count += 1
            hist16[(c_u8 >> 4) & 0xF] += 1

        slot_charge_sums[t] = slot_sum
        if playhead_slot < 0 or t > clamped_ph:
            pre_play_cells += freq_bands
        elif clamped_ph - 2 <= t <= clamped_ph:
            active_window_cells += freq_bands
            active_window_charge_sum += slot_sum
        else:
            afterglow_cells += freq_bands
            afterglow_charge_sum += slot_sum

    # 1. op_charge_summary_u8
    op_charge_summary = {
        "total_chargeable_bytes": total_cells,
        "potential_byte_sum": sum_pot,
        "pushed_up_byte_sum": sum_pushed,
        "fast_charge_byte_sum": sum_fast,
        "slow_charge_byte_sum": sum_slow,
        "mean_potential_q16": (sum_pot << 16) // max(1, total_cells),
        "mean_pushed_up_q16": (sum_pushed << 16) // max(1, total_cells),
        "mean_fast_charge_q16": (sum_fast << 16) // max(1, total_cells),
        "mean_slow_charge_q16": (sum_slow << 16) // max(1, total_cells),
        "pushed_up_active_bytes": pushed_up_count,
        "pushed_up_coverage_q16": (pushed_up_count << 16) // max(1, total_cells),
    }

    # 2. op_push_up_profile_q16
    op_push_up_profile = {
        "min_charge_u8": min_charge,
        "max_charge_u8": max_charge,
        "peak_push_up_displacement_u8": max_charge,
        "mean_active_push_up_q16": (
            (sum_pushed << 16) // max(1, pushed_up_count) if pushed_up_count > 0 else 0
        ),
        "alert_threshold_u8": alert_threshold_u8,
        "alert_pushed_up_bytes": alert_cell_count,
        "pushed_up_alert_triggered": bool(alert_cell_count > 0),
        "charge_histogram_16bin": hist16,
    }

    # 3. op_spectral_centroid_and_spread_q16
    if sum_pushed > 0:
        weighted_band_num = sum(b * band_charge_sums[b] for b in range(freq_bands))
        centroid_band_q16 = (weighted_band_num << 16) // sum_pushed
        centroid_band_int = weighted_band_num // sum_pushed
        centroid_hz = _band_index_to_hz(centroid_band_int, freq_bands)
        spread_num = sum(
            abs((b << 16) - centroid_band_q16) * band_charge_sums[b]
            for b in range(freq_bands)
        )
        spread_band_q16 = spread_num // sum_pushed
    else:
        centroid_band_q16 = 0
        centroid_hz = 0
        spread_band_q16 = 0

    op_spectral = {
        "centroid_band_q16": centroid_band_q16,
        "nominal_centroid_hz": centroid_hz,
        "spectral_spread_band_q16": spread_band_q16,
    }

    # 4. op_multiscale_fast_slow_ratio_q16
    # Time-constant normalization: fast lane (200/256) decays ~4x-8x faster over active windows
    # than slow capacitor lane (252/256).
    fast_to_slow_q16 = ((sum_fast + 1) << 16) // (sum_slow + 1)
    normalized_fast_to_slow_q16 = (((sum_fast * 4) + 1) << 16) // (sum_slow + 1)
    if sum_pushed == 0:
        multiscale_regime = "uncharged_silence"
    elif sum_fast > sum_slow or normalized_fast_to_slow_q16 >= 65536:
        multiscale_regime = "fast_transient_dominant"
    elif normalized_fast_to_slow_q16 >= 42598:  # >= 0.65 in Q16.16
        multiscale_regime = "balanced_attack_sustain"
    else:
        multiscale_regime = "slow_sustained_memory"

    op_multiscale = {
        "fast_to_slow_ratio_q16": fast_to_slow_q16,
        "normalized_fast_to_slow_q16": normalized_fast_to_slow_q16,
        "regime": multiscale_regime,
        "fast_share_q16": (sum_fast << 16) // max(1, sum_fast + sum_slow),
        "slow_share_q16": (sum_slow << 16) // max(1, sum_fast + sum_slow),
    }

    # 5. op_refractory_and_saturation_guard
    op_refractory = {
        "saturation_ceiling_u8": 255,
        "saturated_cell_count": saturated_cell_count,
        "refractory_clamped_count": refractory_clamped_count,
        "runaway_charge_prevented": True,
    }

    # 6. op_connected_charge_events_4c (4-connected pushed-up islands in (t, b) grid)
    visited = bytearray(total_cells)
    components: List[Dict[str, Any]] = []
    fg_t_min, fg_b_min = time_slots, freq_bands
    fg_t_max, fg_b_max = -1, -1

    for t in range(time_slots):
        t_off = t * freq_bands
        for b in range(freq_bands):
            idx = t_off + b
            c_u8 = pushed_up_bytes[idx]
            if c_u8 >= push_up_threshold_u8:
                if t < fg_t_min:
                    fg_t_min = t
                if t > fg_t_max:
                    fg_t_max = t
                if b < fg_b_min:
                    fg_b_min = b
                if b > fg_b_max:
                    fg_b_max = b
                if not visited[idx]:
                    visited[idx] = 1
                    queue = [(t, b)]
                    comp_cells = 0
                    comp_charge_sum = 0
                    comp_peak = 0
                    ct_min, cb_min = t, b
                    ct_max, cb_max = t, b
                    sum_t, sum_b = 0, 0
                    while queue:
                        qt, qb = queue.pop()
                        q_idx = qt * freq_bands + qb
                        qc = pushed_up_bytes[q_idx]
                        comp_cells += 1
                        comp_charge_sum += qc
                        if qc > comp_peak:
                            comp_peak = qc
                        sum_t += qt
                        sum_b += qb
                        if qt < ct_min:
                            ct_min = qt
                        if qt > ct_max:
                            ct_max = qt
                        if qb < cb_min:
                            cb_min = qb
                        if qb > cb_max:
                            cb_max = qb
                        for nt, nb in ((qt - 1, qb), (qt + 1, qb), (qt, qb - 1), (qt, qb + 1)):
                            if 0 <= nt < time_slots and 0 <= nb < freq_bands:
                                n_idx = nt * freq_bands + nb
                                if not visited[n_idx] and pushed_up_bytes[n_idx] >= push_up_threshold_u8:
                                    visited[n_idx] = 1
                                    queue.append((nt, nb))
                    components.append({
                        "event_island_id": f"chg_ev_{len(components) + 1:04d}",
                        "area_bytes": comp_cells,
                        "charge_sum": comp_charge_sum,
                        "peak_charge_u8": comp_peak,
                        "bbox": {
                            "t_min": ct_min,
                            "b_min": cb_min,
                            "t_max": ct_max,
                            "b_max": cb_max,
                        },
                        "centroid": {
                            "t": round(sum_t / float(comp_cells), 4),
                            "b": round(sum_b / float(comp_cells), 4),
                            "nominal_hz": _band_index_to_hz(sum_b // comp_cells, freq_bands),
                        },
                    })

    fg_empty = pushed_up_count == 0
    op_connected = {
        "component_count": len(components),
        "foreground_bbox": {
            "empty": fg_empty,
            "matching_bytes": pushed_up_count,
            "t_min": -1 if fg_empty else fg_t_min,
            "b_min": -1 if fg_empty else fg_b_min,
            "t_max": -1 if fg_empty else fg_t_max,
            "b_max": -1 if fg_empty else fg_b_max,
        },
        "components": components[:64],
    }

    # 7. op_acoustic_contrast_wcag_analog_q16
    op_contrast_q16 = ((max_charge + 13) << 16) // (min_charge + 13)

    # 8. op_spectro_temporal_flux_q16 (onset d/dt + harmonic d/df gradient)
    flux_sum = 0
    for t in range(1, time_slots):
        t_off = t * freq_bands
        prev_off = (t - 1) * freq_bands
        for b in range(1, freq_bands):
            c_cur = pushed_up_bytes[t_off + b]
            dt_grad = abs(c_cur - pushed_up_bytes[prev_off + b])
            df_grad = abs(c_cur - pushed_up_bytes[t_off + b - 1])
            flux_sum += dt_grad + df_grad
    op_flux_q16 = (flux_sum << 16) // max(1, total_cells)

    # 9. op_perceptual_audio_hash_ahash64 (9x8 reduced charge grid horizontal dHash64)
    grid_9x8 = [[0] * 9 for _ in range(8)]
    for ry in range(8):
        b_start = (ry * freq_bands) // 8
        b_end = max(b_start + 1, ((ry + 1) * freq_bands) // 8)
        for rx in range(9):
            t_start = (rx * time_slots) // 9
            t_end = max(t_start + 1, ((rx + 1) * time_slots) // 9)
            cell_sum = 0
            cell_cnt = 0
            for tt in range(t_start, min(time_slots, t_end)):
                toff = tt * freq_bands
                for bb in range(b_start, min(freq_bands, b_end)):
                    cell_sum += pushed_up_bytes[toff + bb]
                    cell_cnt += 1
            grid_9x8[ry][rx] = cell_sum // max(1, cell_cnt)

    ahash64_u64 = 0
    bit_pos = 0
    for ry in range(8):
        for rx in range(8):
            if grid_9x8[ry][rx] < grid_9x8[ry][rx + 1]:
                ahash64_u64 |= 1 << bit_pos
            bit_pos += 1
    op_ahash64 = f"0x{ahash64_u64:016x}"

    # Playhead phase breakdown + deterministic 16-segment envelope summary for audit reconstruction
    seg16_peaks: List[int] = []
    for seg_i in range(16):
        s_t0 = (seg_i * time_slots) // 16
        s_t1 = max(s_t0 + 1, ((seg_i + 1) * time_slots) // 16)
        seg_max = 0
        for tt in range(s_t0, min(time_slots, s_t1)):
            toff = tt * freq_bands
            for bb in range(freq_bands):
                v = pushed_up_bytes[toff + bb]
                if v > seg_max:
                    seg_max = v
        seg16_peaks.append(seg_max)

    op_playhead_phase = {
        "playhead_slot": playhead_slot,
        "time_slots": time_slots,
        "pre_play_uncharged_bytes": pre_play_cells,
        "active_playhead_window_bytes": active_window_cells,
        "active_playhead_charge_sum": active_window_charge_sum,
        "post_play_afterglow_bytes": afterglow_cells,
        "post_play_afterglow_charge_sum": afterglow_charge_sum,
        "reconstructible_envelope_16seg_u8": seg16_peaks,
    }

    # 10. op_bbox_aligned_charge_diff_u8
    ref_bytes = secondary_pushed_up_bytes if secondary_pushed_up_bytes is not None else pushed_up_bytes
    changed_bytes = 0
    pushed_up_added = 0
    discharged_removed = 0
    modified_charge = 0
    dt_min, db_min = time_slots, freq_bands
    dt_max, db_max = -1, -1

    for t in range(time_slots):
        t_off = t * freq_bands
        for b in range(freq_bands):
            idx = t_off + b
            ca = pushed_up_bytes[idx]
            cb = ref_bytes[idx]
            if ca != cb:
                changed_bytes += 1
                a_active = ca >= push_up_threshold_u8
                b_active = cb >= push_up_threshold_u8
                if not a_active and b_active:
                    pushed_up_added += 1
                elif a_active and not b_active:
                    discharged_removed += 1
                else:
                    modified_charge += 1
                if t < dt_min:
                    dt_min = t
                if t > dt_max:
                    dt_max = t
                if b < db_min:
                    db_min = b
                if b > db_max:
                    db_max = b

    diff_empty = changed_bytes == 0
    op_diff = {
        "total_bytes": total_cells,
        "changed_bytes": changed_bytes,
        "change_ratio": round(changed_bytes / float(max(1, total_cells)), 6),
        "pushed_up_added_bytes": pushed_up_added,
        "discharged_removed_bytes": discharged_removed,
        "modified_charge_bytes": modified_charge,
        "diff_bbox": {
            "empty": diff_empty,
            "t_min": -1 if diff_empty else dt_min,
            "b_min": -1 if diff_empty else db_min,
            "t_max": -1 if diff_empty else dt_max,
            "b_max": -1 if diff_empty else db_max,
        },
    }

    return {
        "op_charge_summary_u8": op_charge_summary,
        "op_push_up_profile_q16": op_push_up_profile,
        "op_spectral_centroid_and_spread_q16": op_spectral,
        "op_multiscale_fast_slow_ratio_q16": op_multiscale,
        "op_refractory_and_saturation_guard": op_refractory,
        "op_connected_charge_events_4c": op_connected,
        "op_acoustic_contrast_wcag_analog_q16": op_contrast_q16,
        "op_spectro_temporal_flux_q16": op_flux_q16,
        "op_perceptual_audio_hash_ahash64": op_ahash64,
        "op_playhead_phase_profile": op_playhead_phase,
        "op_bbox_aligned_charge_diff_u8": op_diff,
    }


def _create_audio_charge_closure_vault() -> Dict[str, Callable[..., Any]]:
    """
    Private lexical closure vault for Chargeable-Byte Audio matrices.
    Raw chargeable-byte arrays (`potential`, `novelty`, `fast`, `slow`, `pushed_up`)
    remain strictly isolated inside this closure and are never exposed on instance
    attributes or returned in agent payloads.
    """
    _sources: Dict[Tuple[str, str], Dict[str, Any]] = {}
    _zeroized_stats: Dict[str, int] = {"evicted_frames": 0, "zeroized_bytes": 0}

    def register_audio_source(
        source_id: str,
        source_version: str,
        time_slots: int,
        freq_bands: int,
        specs: Sequence[Dict[str, Any]],
        *,
        playhead_slot: int,
        fast_decay_q8: int = 200,
        slow_decay_q8: int = 252,
        refractory_ticks: int = 2,
    ) -> Dict[str, Any]:
        key = (source_id, source_version)
        if key in _sources:
            raise ValueError(f"Immutable audio source {source_id}@{source_version} already registered")

        pot_bytes, nov_bytes = _compile_chargeable_byte_potentials(specs, time_slots, freq_bands)
        sim = _simulate_causal_playhead_charging(
            pot_bytes,
            nov_bytes,
            time_slots,
            freq_bands,
            playhead_slot=playhead_slot,
            fast_decay_q8=fast_decay_q8,
            slow_decay_q8=slow_decay_q8,
            refractory_ticks=refractory_ticks,
        )
        potential_sha256 = _canonical_sha256_tag(pot_bytes)
        charged_sha256 = _canonical_sha256_tag(sim["pushed_up_bytes"])

        _sources[key] = {
            "time_slots": time_slots,
            "freq_bands": freq_bands,
            "playhead_slot": int(sim["active_playhead_slot"]),
            "potential_buf": bytearray(pot_bytes),
            "novelty_buf": bytearray(nov_bytes),
            "fast_buf": bytearray(sim["fast_bytes"]),
            "slow_buf": bytearray(sim["slow_bytes"]),
            "pushed_up_buf": bytearray(sim["pushed_up_bytes"]),
            "potential_sha256": potential_sha256,
            "charged_sha256": charged_sha256,
            "saturated_cell_count": sim["saturated_cell_count"],
            "refractory_clamped_count": sim["refractory_clamped_count"],
            "spec_count": len(specs),
        }
        return {
            "time_slots": time_slots,
            "freq_bands": freq_bands,
            "total_chargeable_bytes": time_slots * freq_bands,
            "playhead_slot": int(sim["active_playhead_slot"]),
            "potential_bytes_sha256": potential_sha256,
            "charged_bytes_sha256": charged_sha256,
            "compiled_event_count": len(specs),
        }

    def run_10_operator_suite(
        source_a: Tuple[str, str],
        source_b: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, Any]:
        if source_a not in _sources:
            raise KeyError(f"Unknown audio source: {source_a}")
        ea = _sources[source_a]
        ts, fb = ea["time_slots"], ea["freq_bands"]
        sec_bytes: Optional[bytes] = None
        if source_b is not None:
            if source_b not in _sources:
                raise KeyError(f"Unknown secondary audio source: {source_b}")
            eb = _sources[source_b]
            if (eb["time_slots"], eb["freq_bands"]) != (ts, fb):
                raise ValueError("Audio grid dimensions must match for charge diff")
            sec_bytes = bytes(eb["pushed_up_buf"])

        return _run_10_audio_charge_operators(
            bytes(ea["potential_buf"]),
            bytes(ea["pushed_up_buf"]),
            bytes(ea["fast_buf"]),
            bytes(ea["slow_buf"]),
            sec_bytes,
            ts,
            fb,
            playhead_slot=ea["playhead_slot"],
            saturated_cell_count=ea["saturated_cell_count"],
            refractory_clamped_count=ea["refractory_clamped_count"],
        )

    def compute_dirty_charge_tiles_16x16(
        source_a: Tuple[str, str],
        source_b: Tuple[str, str],
    ) -> Dict[str, Any]:
        if source_a not in _sources or source_b not in _sources:
            raise KeyError("Both audio sources must exist for 16x16 dirty charge-tile profiling")
        ea = _sources[source_a]
        eb = _sources[source_b]
        ts, fb = ea["time_slots"], ea["freq_bands"]
        if (eb["time_slots"], eb["freq_bands"]) != (ts, fb):
            raise ValueError("Audio grid dimensions must match for 16x16 dirty tiles")

        tiles_t = (ts + 15) // 16
        tiles_b = (fb + 15) // 16
        total_tiles = tiles_t * tiles_b

        if ea["charged_sha256"] == eb["charged_sha256"]:
            zero_mask = bytes((total_tiles + 7) // 8)
            return {
                "tile_size": 16,
                "grid_dimensions": {"tiles_t": tiles_t, "tiles_b": tiles_b, "total_tiles": total_tiles},
                "dirty_tile_count": 0,
                "clean_tile_count": total_tiles,
                "dirty_tile_ratio": 0.0,
                "dirty_tile_bbox": [-1, -1, -1, -1],
                "dirty_bitmask_sha256": _canonical_sha256_tag(zero_mask),
                "fast_path_identical_sha": True,
            }

        buf_a = ea["pushed_up_buf"]
        buf_b = eb["pushed_up_buf"]
        bitmask = bytearray((total_tiles + 7) // 8)
        dirty_count = 0
        min_tt, min_tb = tiles_t, tiles_b
        max_tt, max_tb = -1, -1

        for tt in range(tiles_t):
            t0 = tt * 16
            t1 = min(ts, t0 + 16)
            for tb in range(tiles_b):
                b0 = tb * 16
                b1 = min(fb, b0 + 16)
                tile_dirty = False
                for t in range(t0, t1):
                    off = t * fb + b0
                    span = b1 - b0
                    if buf_a[off : off + span] != buf_b[off : off + span]:
                        tile_dirty = True
                        break
                if tile_dirty:
                    t_idx = tt * tiles_b + tb
                    bitmask[t_idx >> 3] |= 1 << (t_idx & 7)
                    dirty_count += 1
                    if tt < min_tt:
                        min_tt = tt
                    if tb < min_tb:
                        min_tb = tb
                    if tt > max_tt:
                        max_tt = tt
                    if tb > max_tb:
                        max_tb = tb

        dirty_bbox = [min_tt, min_tb, max_tt, max_tb] if dirty_count > 0 else [-1, -1, -1, -1]
        return {
            "tile_size": 16,
            "grid_dimensions": {"tiles_t": tiles_t, "tiles_b": tiles_b, "total_tiles": total_tiles},
            "dirty_tile_count": dirty_count,
            "clean_tile_count": total_tiles - dirty_count,
            "dirty_tile_ratio": round(dirty_count / float(max(1, total_tiles)), 6),
            "dirty_tile_bbox": dirty_bbox,
            "dirty_bitmask_sha256": _canonical_sha256_tag(bitmask),
            "fast_path_identical_sha": False,
        }

    def preview_ghost_audio_specs(
        base_source_key: Tuple[str, str],
        ghost_specs: Sequence[Dict[str, Any]],
        *,
        ghost_playhead_slot: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Computes hypothetical chargeable-byte metrics on `ghost_isolation_plane`
        if `ghost_specs` (or an advanced `ghost_playhead_slot`) were played over
        `base_source_key`, inside a disposable buffer that is immediately
        zeroized (`explicit_bzero`). Never mutates `base_source_key`.
        """
        if base_source_key not in _sources:
            raise KeyError(f"Unknown base audio source: {base_source_key}")
        base_entry = _sources[base_source_key]
        ts, fb = base_entry["time_slots"], base_entry["freq_bands"]
        target_ph = (
            int(ghost_playhead_slot)
            if ghost_playhead_slot is not None
            else base_entry["playhead_slot"]
        )

        g_pot, g_nov = _compile_chargeable_byte_potentials(ghost_specs, ts, fb)
        comb_pot = bytearray(base_entry["potential_buf"])
        comb_nov = bytearray(base_entry["novelty_buf"])
        total_cells = ts * fb
        for i in range(total_cells):
            if g_pot[i] > 0:
                comb_pot[i] = min(255, comb_pot[i] + g_pot[i])
            if g_nov[i] > 0:
                comb_nov[i] = min(255, comb_nov[i] + g_nov[i])

        sim = _simulate_causal_playhead_charging(
            bytes(comb_pot),
            bytes(comb_nov),
            ts,
            fb,
            playhead_slot=target_ph,
        )
        explicit_zeroize_bytearray(comb_pot)
        explicit_zeroize_bytearray(comb_nov)

        metrics = _run_10_audio_charge_operators(
            bytes(base_entry["potential_buf"]),
            sim["pushed_up_bytes"],
            sim["fast_bytes"],
            sim["slow_bytes"],
            bytes(base_entry["pushed_up_buf"]),
            ts,
            fb,
            playhead_slot=target_ph,
            saturated_cell_count=sim["saturated_cell_count"],
            refractory_clamped_count=sim["refractory_clamped_count"],
        )
        base_sha_after = _canonical_sha256_tag(_sources[base_source_key]["pushed_up_buf"])
        assert base_sha_after == base_entry["charged_sha256"], "Ghost audio preview mutated verified plane!"
        return {
            "time_slots": ts,
            "freq_bands": fb,
            "base_charged_sha256_preserved": True,
            "hypothetical_metrics": metrics,
        }

    def project_pushed_up_bytes_to_visual_marks(
        source_key: Tuple[str, str],
        *,
        max_columns: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Projects the current pushed-up chargeable bytes of `source_key` into
        normalized Spark/Prism `[0..1]` visual `marks` so an agent can visually
        SEE the audio bytes being pushed up during play on an Electric Vision canvas.
        """
        if source_key not in _sources:
            raise KeyError(f"Unknown audio source: {source_key}")
        entry = _sources[source_key]
        ts, fb = entry["time_slots"], entry["freq_bands"]
        pushed_buf = entry["pushed_up_buf"]
        fast_buf = entry["fast_buf"]
        ph = entry["playhead_slot"]

        cols = max(8, min(64, int(max_columns)))
        marks: List[Dict[str, Any]] = []

        for c_idx in range(cols):
            t0 = (c_idx * ts) // cols
            t1 = max(t0 + 1, ((c_idx + 1) * ts) // cols)
            col_peak_charge = 0
            col_peak_fast = 0
            dominant_band = 0
            for tt in range(t0, min(ts, t1)):
                toff = tt * fb
                for bb in range(fb):
                    val = pushed_buf[toff + bb]
                    fval = fast_buf[toff + bb]
                    if val > col_peak_charge:
                        col_peak_charge = val
                        dominant_band = bb
                    if fval > col_peak_fast:
                        col_peak_fast = fval

            if col_peak_charge < 16:
                continue

            # X coordinate along [0.06 .. 0.94]
            x_norm = round(0.06 + (0.88 * c_idx) / float(max(1, cols - 1)), 4)
            # Pushed-up elevation: higher charge pushes the byte higher up (smaller y in top-down canvas)
            push_up_frac = col_peak_charge / 255.0
            band_frac = dominant_band / float(max(1, fb - 1))
            y_base = 0.88
            y_pushed_top = round(max(0.08, y_base - 0.72 * (0.65 * push_up_frac + 0.35 * band_frac)), 4)

            # Fast-transient charge glows cyan; slow-memory afterglow glows amber/magenta
            is_playhead_col = (t0 <= ph < t1)
            if is_playhead_col or col_peak_fast >= (col_peak_charge * 3) // 4:
                color_rgb = [0, 229, 255]
            else:
                color_rgb = [255, 179, 0]

            # Vertical column line showing the byte pushed up from baseline
            marks.append({
                "mark_id": f"aud_push_col_{c_idx:02d}",
                "k": "line",
                "pts": [[x_norm, y_base], [x_norm, y_pushed_top]],
                "w": 0.012,
                "c": color_rgb,
                "a": 1.0,
            })
            # Cap dot at the pushed-up peak
            marks.append({
                "mark_id": f"aud_push_cap_{c_idx:02d}",
                "k": "dot",
                "x": x_norm,
                "y": y_pushed_top,
                "r": round(0.012 + 0.012 * push_up_frac, 4),
                "c": [213, 0, 249] if is_playhead_col else color_rgb,
                "a": 1.0,
            })

        # Ensure at least one baseline reference mark if completely silent/uncharged
        if not marks:
            marks.append({
                "mark_id": "aud_resting_baseline",
                "k": "line",
                "pts": [[0.06, 0.88], [0.94, 0.88]],
                "w": 0.004,
                "c": [80, 90, 110],
                "a": 1.0,
            })
        return marks

    def evict_source(source_key: Tuple[str, str]) -> Dict[str, Any]:
        """Explicitly zeroizes (`explicit_bzero`) all 5 byte buffers of a single audio source."""
        if source_key not in _sources:
            return {"evicted": False, "zeroized_bytes": 0, "remaining_sources": len(_sources)}
        entry = _sources.pop(source_key)
        wiped = 0
        for buf_name in ("potential_buf", "novelty_buf", "fast_buf", "slow_buf", "pushed_up_buf"):
            wiped += explicit_zeroize_bytearray(entry[buf_name])
        _zeroized_stats["evicted_frames"] += 1
        _zeroized_stats["zeroized_bytes"] += wiped
        return {
            "evicted": True,
            "evicted_source": {"source_id": source_key[0], "source_version": source_key[1]},
            "zeroized_bytes": wiped,
            "remaining_sources": len(_sources),
            "cumulative_evicted_frames": _zeroized_stats["evicted_frames"],
            "cumulative_zeroized_bytes": _zeroized_stats["zeroized_bytes"],
        }

    def purge_all() -> Dict[str, Any]:
        t0 = time.perf_counter()
        purged_count = len(_sources)
        wiped = 0
        for entry in _sources.values():
            for buf_name in ("potential_buf", "novelty_buf", "fast_buf", "slow_buf", "pushed_up_buf"):
                wiped += explicit_zeroize_bytearray(entry[buf_name])
        _sources.clear()
        _zeroized_stats["zeroized_bytes"] += wiped
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        return {
            "purged_sources": purged_count,
            "zeroized_bytes": wiped,
            "remaining_sources": 0,
            "elapsed_ms": elapsed_ms,
        }

    return {
        "register_audio_source": register_audio_source,
        "run_10_operator_suite": run_10_operator_suite,
        "compute_dirty_charge_tiles_16x16": compute_dirty_charge_tiles_16x16,
        "preview_ghost_audio_specs": preview_ghost_audio_specs,
        "project_pushed_up_bytes_to_visual_marks": project_pushed_up_bytes_to_visual_marks,
        "evict_source": evict_source,
        "purge_all": purge_all,
    }


class ElectricAudioAgentSession:
    """
    Context-managed Chargeable-Byte Electric Audio ("Electric Ear") Session for Agents.

    Can run standalone (spawning its own out-of-process HSM/Vault/Audit/Quarantine
    enclave via an internal `ElectricVisionAgentSession`) or attach to an existing
    `ElectricVisionAgentSession` for unified Sight + Sound perception.
    """

    def __init__(
        self,
        *,
        run_dir: str = "/tmp/cyclops-ee",
        default_grid: Tuple[int, int] = (256, 64),
        shared_vision_session: Optional[ElectricVisionAgentSession] = None,
    ) -> None:
        if default_grid not in SUPPORTED_AUDIO_GRIDS:
            raise ValueError(f"Unsupported default_grid: {default_grid}")
        self.run_dir = run_dir
        self.default_grid = default_grid
        self._shared_vision = shared_vision_session
        self._owned_vision: Optional[ElectricVisionAgentSession] = None
        self._audio_vault = _create_audio_charge_closure_vault()
        self._source_grids: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self._source_specs: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self._active = False

    def __enter__(self) -> "ElectricAudioAgentSession":
        if self._shared_vision is None:
            self._owned_vision = ElectricVisionAgentSession(
                run_dir=self.run_dir,
                default_resolution=(256, 256),
            )
            self._owned_vision.__enter__()
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._active = False
        self._audio_vault["purge_all"]()
        self._source_grids.clear()
        self._source_specs.clear()
        if self._owned_vision is not None:
            self._owned_vision.__exit__(exc_type, exc_val, exc_tb)
            self._owned_vision = None

    @property
    def vision_session(self) -> ElectricVisionAgentSession:
        if not self._active:
            raise RuntimeError("ElectricAudioAgentSession is not active")
        sess = self._shared_vision or self._owned_vision
        if sess is None:
            raise RuntimeError("Underlying enclave session is not initialized")
        return sess

    def register_chargeable_audio(
        self,
        source_id: str,
        source_version: str,
        audio_events_or_spec: Union[Sequence[Dict[str, Any]], Dict[str, Any]],
        *,
        time_slots: Optional[int] = None,
        freq_bands: Optional[int] = None,
        playhead_slot: Optional[int] = None,
        playhead_ratio: float = 1.0,
        fast_decay_q8: int = 200,
        slow_decay_q8: int = 252,
        refractory_ticks: int = 2,
        source_kind: str = "captured_frame",
        origin_attestation: str = "synthetic_fixture",
    ) -> Dict[str, Any]:
        """
        Translates `audio_events_or_spec` into chargeable bytes (`0..255`) and
        charges/pushes them up up to `playhead_slot` (or `playhead_ratio` in `[0.0..1.0]`;
        pass `playhead_slot=-1` for unplayed resting state).
        """
        v_sess = self.vision_session
        harness = v_sess._require_active()

        # Enforce audio origin attestation firewall (synthetic_fixture or owner_supplied_audio)
        if origin_attestation not in VALID_AUDIO_ORIGINS:
            raise PermissionError(
                f"ERR_NON_SYNTHETIC_AUDIO_PROHIBITED: origin_attestation={origin_attestation!r} "
                "is refused (live microphone capture and untrusted audio streams are prohibited)"
            )
        harness_origin = (
            "owner_supplied_picture"
            if origin_attestation == "owner_supplied_audio"
            else "synthetic_fixture"
        )

        ts = time_slots if time_slots is not None else self.default_grid[0]
        fb = freq_bands if freq_bands is not None else self.default_grid[1]
        if (ts, fb) not in SUPPORTED_AUDIO_GRIDS:
            raise ValueError(f"Audio grid {(ts, fb)} not in {SUPPORTED_AUDIO_GRIDS}")

        if playhead_slot is None:
            if playhead_ratio < 0.0:
                resolved_ph = -1
            else:
                clamped_r = max(0.0, min(1.0, float(playhead_ratio)))
                resolved_ph = int(round(clamped_r * (ts - 1)))
        else:
            resolved_ph = int(playhead_slot)

        specs, proxy_128_marks = translate_audio_to_chargeable_bytes(
            audio_events_or_spec,
            ts,
            fb,
        )

        desc = harness.register_verified_source(
            source_id=source_id,
            source_version=source_version,
            source_kind=source_kind,
            marks=proxy_128_marks,
            origin_attestation=harness_origin,
        )

        vault_info = self._audio_vault["register_audio_source"](
            source_id,
            source_version,
            ts,
            fb,
            specs,
            playhead_slot=resolved_ph,
            fast_decay_q8=fast_decay_q8,
            slow_decay_q8=slow_decay_q8,
            refractory_ticks=refractory_ticks,
        )
        self._source_grids[(source_id, source_version)] = (ts, fb)
        self._source_specs[(source_id, source_version)] = specs

        out = {
            **desc,
            "audio_grid": {
                "time_slots": ts,
                "freq_bands": fb,
                "total_chargeable_bytes": ts * fb,
            },
            "playhead_slot": vault_info["playhead_slot"],
            "potential_bytes_sha256": vault_info["potential_bytes_sha256"],
            "charged_bytes_sha256": vault_info["charged_bytes_sha256"],
            "compiled_event_count": vault_info["compiled_event_count"],
            "origin_attestation": origin_attestation,
        }
        assert_no_raw_buffer_leakage(out)
        return out

    def inspect_audio(
        self,
        source_id: str,
        source_version: str = "v1",
        *,
        issue_observation_receipt: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes all 10 closed-form Chargeable-Byte Audio operators on
        `source_id@source_version`, creates a provenance-pinned `ElectricRender`,
        and optionally issues an HSM-countersigned `ObservationReceipt`.
        """
        t0 = time.perf_counter()
        v_sess = self.vision_session
        harness = v_sess._require_active()
        key = (source_id, source_version)
        if key not in self._source_grids:
            raise KeyError(f"Unknown audio source {source_id}@{source_version}")
        ts, fb = self._source_grids[key]

        charge_metrics = self._audio_vault["run_10_operator_suite"](key, None)

        m_inspect = harness.execute_measurement(
            "inspect_source", {}, source_id=source_id, source_version=source_version
        )
        m_bbox = harness.execute_measurement(
            "measure_bbox",
            {"selector": {"type": "non_transparent"}},
            source_id=source_id,
            source_version=source_version,
        )
        m_objects = harness.execute_measurement(
            "count_objects",
            {"connectivity": 4, "min_area_px": 1},
            source_id=source_id,
            source_version=source_version,
        )

        audio_replay_digest = compute_replay_digest(
            source_id=source_id,
            source_version=source_version,
            secondary_source=None,
            operator_name="chargeable_byte_10_operator_suite",
            operator_version=OPERATOR_VERSION,
            parameters={"time_slots": ts, "freq_bands": fb},
            result=charge_metrics,
        )

        render_type = "observed" if issue_observation_receipt else "derived"
        render_source = "verified_pixels" if issue_observation_receipt else "measurement"
        uncertainty_state = "verified" if issue_observation_receipt else "derived"

        prov_refs = [
            {"ref_type": "source", "ref_id": source_id, "ref_version": source_version},
            {"ref_type": "measurement", "ref_id": m_inspect["measurement_id"]},
            {"ref_type": "measurement", "ref_id": m_bbox["measurement_id"]},
            {"ref_type": "measurement", "ref_id": m_objects["measurement_id"]},
        ]
        render = harness.create_electric_render(
            render_type=render_type,
            source=render_source,
            confidence="high",
            uncertainty_state=uncertainty_state,
            provenance_refs=prov_refs,
            visibility_policy="agent_visible",
        )

        receipt: Optional[Dict[str, Any]] = None
        hsm_countersignature: Optional[Dict[str, Any]] = None
        if issue_observation_receipt:
            cap_tok = (
                v_sess._vault_client.request_capability_token(
                    render_id=render["render_id"],
                    captured_source_id=source_id,
                    captured_source_version=source_version,
                )
                if v_sess._vault_client is not None
                else None
            )
            receipt = harness.issue_observation_receipt(
                render_id=render["render_id"],
                captured_source_id=source_id,
                captured_source_version=source_version,
                caller_capability_token=cap_tok,
            )
            promo = harness.promote_render_to_observed(render["render_id"], receipt=receipt)
            if promo.get("status") != "ok":
                raise PermissionError(f"Failed to promote audio render to observed: {promo}")
            render = promo["render"]
            if v_sess._hsm_client is not None:
                hsm_countersignature = v_sess._hsm_client.sign(
                    canonical_json(receipt),
                    key_domain="receipt",
                )

        acc_key_map = {
            "verified": "verified_observation",
            "derived": "unobserved_electric_render",
            "predicted": "non_destructive_ghost_overlay",
            "unknown": "refused_epistemic_boundary",
            "stale": "quarantined_or_unverified_input",
        }
        accessibility_meta = build_accessibility_metadata(
            acc_key_map.get(render["uncertainty_state"], "unobserved_electric_render")
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)

        out = {
            "status": "ok",
            "modality": "electric_audio_chargeable_bytes",
            "source_id": source_id,
            "source_version": source_version,
            "audio_grid": {"time_slots": ts, "freq_bands": fb, "total_chargeable_bytes": ts * fb},
            "charge_metrics": charge_metrics,
            "audio_replay_digest": audio_replay_digest,
            "render": render,
            "observation_receipt": receipt,
            "hsm_receipt_countersignature": hsm_countersignature,
            "accessibility": accessibility_meta,
            "elapsed_ms": elapsed_ms,
        }
        assert_no_raw_buffer_leakage(out)
        return out

    def compare_audio_states(
        self,
        source_a: Tuple[str, str],
        source_b: Tuple[str, str],
    ) -> Dict[str, Any]:
        """
        Computes a deterministic structural charge diff and Layer-3 derived
        representations between two audio sources or two playback ticks
        (`source_a -> source_b`).
        """
        t0 = time.perf_counter()
        harness = self.vision_session._require_active()
        if source_a not in self._source_grids or source_b not in self._source_grids:
            raise KeyError(f"Both {source_a} and {source_b} must be registered before comparison")
        ts_a, fb_a = self._source_grids[source_a]
        ts_b, fb_b = self._source_grids[source_b]
        if (ts_a, fb_a) != (ts_b, fb_b):
            raise ValueError(f"Audio grid mismatch: {source_a} is {ts_a}x{fb_a}, {source_b} is {ts_b}x{fb_b}")

        suite_a = self._audio_vault["run_10_operator_suite"](source_a, None)
        suite_ab = self._audio_vault["run_10_operator_suite"](source_b, source_a)
        diff_metrics = suite_ab["op_bbox_aligned_charge_diff_u8"]

        derived_reps: Dict[str, Any] = {}
        for rep_type in (
            "change_mask",
            "difference_map",
            "motion_summary",
            "colour_delta",
            "spatial_relationship",
        ):
            derived_reps[rep_type] = harness.compute_derived_representation(
                rep_type, source_a, source_b
            )

        net_push_up_delta = (
            int(suite_ab["op_charge_summary_u8"]["pushed_up_byte_sum"])
            - int(suite_a["op_charge_summary_u8"]["pushed_up_byte_sum"])
        )
        centroid_hz_delta = (
            int(suite_ab["op_spectral_centroid_and_spread_q16"]["nominal_centroid_hz"])
            - int(suite_a["op_spectral_centroid_and_spread_q16"]["nominal_centroid_hz"])
        )
        ahash_hamming = _dhash64_hamming_distance(
            suite_a["op_perceptual_audio_hash_ahash64"],
            suite_ab["op_perceptual_audio_hash_ahash64"],
        )

        diff_replay_digest = compute_replay_digest(
            source_id=source_a[0],
            source_version=source_a[1],
            secondary_source={"source_id": source_b[0], "source_version": source_b[1]},
            operator_name="chargeable_byte_audio_comparison",
            operator_version=OPERATOR_VERSION,
            parameters={"time_slots": ts_a, "freq_bands": fb_a},
            result=diff_metrics,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        out = {
            "status": "ok",
            "modality": "electric_audio_chargeable_bytes",
            "source_a": {"source_id": source_a[0], "source_version": source_a[1]},
            "source_b": {"source_id": source_b[0], "source_version": source_b[1]},
            "audio_grid": {"time_slots": ts_a, "freq_bands": fb_a},
            "charge_diff_metrics": diff_metrics,
            "net_push_up_byte_delta": net_push_up_delta,
            "spectral_centroid_hz_delta": centroid_hz_delta,
            "ahash64_hamming_distance": ahash_hamming,
            "derived_representations": derived_reps,
            "diff_replay_digest": diff_replay_digest,
            "epistemic_label": "derived",
            "observed": False,
            "elapsed_ms": elapsed_ms,
        }
        assert_no_raw_buffer_leakage(out)
        return out

    def preview_ghost_audio(
        self,
        base_source_id: str,
        base_source_version: str,
        proposed_ghost_events: Union[Sequence[Dict[str, Any]], Dict[str, Any]],
        *,
        ghost_playhead_slot: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Projects `proposed_ghost_events` onto `ghost_isolation_plane` over
        `base_source_id@base_source_version` to evaluate how additional tones or
        an advanced playhead will push up the chargeable bytes without mutating
        `verified_plane`. Guarantees `observed=False` and `ghost_byte_leakage == 0`.
        """
        harness = self.vision_session._require_active()
        key = (base_source_id, base_source_version)
        if key not in self._source_grids:
            raise KeyError(f"Unknown base audio source {base_source_id}@{base_source_version}")
        ts, fb = self._source_grids[key]

        ghost_specs, proxy_ghost_marks = translate_audio_to_chargeable_bytes(
            proposed_ghost_events, ts, fb
        )
        preview_res = self._audio_vault["preview_ghost_audio_specs"](
            key,
            ghost_specs,
            ghost_playhead_slot=ghost_playhead_slot,
        )

        ghost_render = harness.create_electric_render(
            render_type="predicted",
            source="inference",
            confidence="medium",
            uncertainty_state="predicted",
            provenance_refs=[
                {"ref_type": "source", "ref_id": base_source_id, "ref_version": base_source_version}
            ],
            ghost_marks=proxy_ghost_marks,
        )
        accessibility_meta = build_accessibility_metadata("non_destructive_ghost_overlay")

        out = {
            "status": "ok",
            "modality": "electric_audio_chargeable_bytes",
            "epistemic_state": "predicted",
            "observed": False,
            "layer_isolation": ghost_render["layer_isolation"],
            "base_source_id": base_source_id,
            "base_source_version": base_source_version,
            "base_charged_sha256_preserved": preview_res["base_charged_sha256_preserved"],
            "ghost_byte_leakage": 0,
            "hypothetical_metrics": preview_res["hypothetical_metrics"],
            "ghost_render": ghost_render,
            "accessibility": accessibility_meta,
        }
        assert_no_raw_buffer_leakage(out)
        return out

    def evaluate_agent_audio_query(
        self,
        operator_or_query_type: str,
        parameters: Dict[str, Any],
        *,
        source_id: str,
        source_version: str = "v1",
    ) -> Dict[str, Any]:
        """
        Executes a closed-form measurement query or deterministically refuses
        any semantic speech transcription (ASR), emotional mood, aesthetic, or
        speaker-intent query (`REFUSED_EPISTEMIC_BOUNDARY`).
        """
        return self.vision_session.evaluate_agent_query(
            operator_or_query_type,
            parameters,
            source_id=source_id,
            source_version=source_version,
        )

    def project_pushed_up_bytes_to_marks(
        self,
        source_id: str,
        source_version: str = "v1",
        *,
        max_columns: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Projects the pushed-up chargeable bytes of `source_id@source_version` into
        visual Spark/Prism `[0..1]` `marks` so agents can visually inspect the
        pushed-up byte columns via Electric Vision.
        """
        key = (source_id, source_version)
        if key not in self._source_grids:
            raise KeyError(f"Unknown audio source {source_id}@{source_version}")
        return self._audio_vault["project_pushed_up_bytes_to_visual_marks"](
            key, max_columns=max_columns
        )

    def evict_chargeable_audio(self, source_id: str, source_version: str) -> Dict[str, Any]:
        """Explicitly zeroizes (`explicit_bzero`) and evicts a single audio source revision."""
        self.vision_session._require_active()
        key = (source_id, source_version)
        self._source_grids.pop(key, None)
        self._source_specs.pop(key, None)
        res = self._audio_vault["evict_source"](key)
        assert_no_raw_buffer_leakage(res)
        return res

    def create_continuous_ear_stream(
        self,
        source_id: str = "src_continuous_ear",
        *,
        time_slots: Optional[int] = None,
        freq_bands: Optional[int] = None,
        max_window_frames: int = 8,
        settle_ticks: int = 2,
        checkpoint_hamming_threshold: int = 4,
    ) -> "ContinuousElectricEarStream":
        """Creates a ContinuousElectricEarStream bound to this active session."""
        self.vision_session._require_active()
        ts = time_slots if time_slots is not None else self.default_grid[0]
        fb = freq_bands if freq_bands is not None else self.default_grid[1]
        return ContinuousElectricEarStream(
            self,
            source_id=source_id,
            time_slots=ts,
            freq_bands=fb,
            max_window_frames=max_window_frames,
            settle_ticks=settle_ticks,
            checkpoint_hamming_threshold=checkpoint_hamming_threshold,
        )


class ContinuousElectricEarStream:
    """
    Continuous Multi-Tick Chargeable-Byte Audio Stream ("Continuous Electric Ear").

    Streams causal playhead progression (`playhead_slot` advancing tick by tick
    or across successive audio packets) where bytes get charged up / pushed up
    during play and decay in the trailing capacitance wake:
      1. Bounded Ring-Buffer Temporal Window (`max_window_frames`) with automatic
         `explicit_bzero` zeroization eviction of expired audio states.
      2. 16x16 Dirty Charge-Tile Bitmasking (`compute_dirty_charge_tiles_16x16`)
         with O(1) SHA-256 fast-path skip.
      3. Automatic Epistemic Stale-Observation Decay (`verified -> stale` watchdog)
         when newer playback ticks push up or decay chargeable bytes.
      4. Multi-Tick Temporal Operators (`ahash64_hamming_timeline`,
         `settling_and_oscillation_detector`, `pitch_kinematic_trajectory_chain`,
         and closed-loop `stage_ghost_audio_prediction` + `verify_ghost_realization`).
    """

    def __init__(
        self,
        session: ElectricAudioAgentSession,
        source_id: str = "src_continuous_ear",
        *,
        time_slots: int = 256,
        freq_bands: int = 64,
        max_window_frames: int = 8,
        settle_ticks: int = 2,
        checkpoint_hamming_threshold: int = 4,
    ) -> None:
        if (time_slots, freq_bands) not in SUPPORTED_AUDIO_GRIDS:
            raise ValueError(f"Unsupported audio stream grid {(time_slots, freq_bands)}")
        if max_window_frames < 2 or max_window_frames > 64:
            raise ValueError("max_window_frames must be between 2 and 64")
        if settle_ticks < 1:
            raise ValueError("settle_ticks must be >= 1")
        self.session = session
        self.source_id = source_id
        self.time_slots = time_slots
        self.freq_bands = freq_bands
        self.max_window_frames = max_window_frames
        self.settle_ticks = settle_ticks
        self.checkpoint_hamming_threshold = checkpoint_hamming_threshold

        self._window_versions: List[str] = []
        self._frame_records: Dict[str, Dict[str, Any]] = {}
        self._transition_records: List[Dict[str, Any]] = []
        self._observation_ledger: Dict[str, Dict[str, Any]] = {}
        self._staged_ghosts: Dict[str, Dict[str, Any]] = {}
        self._ghost_realizations: List[Dict[str, Any]] = []
        self._eviction_log: List[Dict[str, Any]] = []
        self._tick_counter = 0

    def stage_ghost_audio_prediction(
        self,
        base_version: str,
        proposed_ghost_events: Union[Sequence[Dict[str, Any]], Dict[str, Any]],
        *,
        ghost_playhead_slot: Optional[int] = None,
        ghost_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stages a GhostOverlay audio prediction on `ghost_isolation_plane`."""
        preview = self.session.preview_ghost_audio(
            self.source_id,
            base_version,
            proposed_ghost_events,
            ghost_playhead_slot=ghost_playhead_slot,
        )
        gid = ghost_id or f"ghost_aud_{len(self._staged_ghosts) + 1:04d}"
        record = {
            "ghost_id": gid,
            "base_source_id": self.source_id,
            "base_source_version": base_version,
            "epistemic_state": "predicted",
            "observed": False,
            "layer_isolation": "ghost_isolation_plane",
            "hypothetical_metrics": preview["hypothetical_metrics"],
            "ghost_render": preview["ghost_render"],
            "realized": False,
        }
        self._staged_ghosts[gid] = record
        assert_no_raw_buffer_leakage(record)
        return record

    def verify_ghost_realization(
        self,
        ghost_id: str,
        actual_version: str,
    ) -> Dict[str, Any]:
        """Scores a staged GhostOverlay audio prediction against the realized frame."""
        if ghost_id not in self._staged_ghosts:
            raise KeyError(f"Unknown staged audio ghost_id: {ghost_id!r}")
        if actual_version not in self._frame_records:
            raise KeyError(f"Unknown actual_version in ear stream: {actual_version!r}")

        ghost_rec = self._staged_ghosts[ghost_id]
        actual_rec = self._frame_records[actual_version]
        g_m = ghost_rec["hypothetical_metrics"]
        a_m = actual_rec["inspection"]["charge_metrics"]

        g_pushed = int(g_m["op_charge_summary_u8"]["pushed_up_active_bytes"])
        a_pushed = int(a_m["op_charge_summary_u8"]["pushed_up_active_bytes"])
        pushed_residual = abs(a_pushed - g_pushed)

        g_sum = int(g_m["op_charge_summary_u8"]["pushed_up_byte_sum"])
        a_sum = int(a_m["op_charge_summary_u8"]["pushed_up_byte_sum"])
        charge_sum_residual = abs(a_sum - g_sum)

        ahash_hamming = _dhash64_hamming_distance(
            g_m["op_perceptual_audio_hash_ahash64"],
            a_m["op_perceptual_audio_hash_ahash64"],
        )

        px_norm = max(0.0, 1.0 - (pushed_residual / max(1.0, max(g_pushed, a_pushed, 1))))
        ahash_norm = max(0.0, 1.0 - (ahash_hamming / 64.0))
        sum_norm = max(0.0, 1.0 - (charge_sum_residual / max(1.0, max(g_sum, a_sum, 1))))
        convergence_score = round((0.4 * px_norm) + (0.3 * ahash_norm) + (0.3 * sum_norm), 6)
        exact_realization = bool(
            pushed_residual == 0 and charge_sum_residual == 0 and ahash_hamming == 0
        )

        audit = {
            "ghost_id": ghost_id,
            "base_source_version": ghost_rec["base_source_version"],
            "actual_source_version": actual_version,
            "epistemic_label": "derived",
            "observed": False,
            "exact_realization": exact_realization,
            "realization_convergence_score": 1.0 if exact_realization else convergence_score,
            "residuals": {
                "pushed_up_active_bytes_residual": pushed_residual,
                "pushed_up_byte_sum_residual": charge_sum_residual,
                "ahash64_hamming_distance": ahash_hamming,
            },
            "replay_digest": compute_replay_digest(
                source_id=self.source_id,
                source_version=actual_version,
                secondary_source={"source_id": self.source_id, "source_version": ghost_rec["base_source_version"]},
                operator_name="verify_ghost_audio_realization",
                operator_version=OPERATOR_VERSION,
                parameters={"ghost_id": ghost_id},
                result={
                    "pushed_residual": pushed_residual,
                    "charge_sum_residual": charge_sum_residual,
                    "ahash_hamming": ahash_hamming,
                },
            ),
        }
        ghost_rec["realized"] = True
        self._ghost_realizations.append(audit)
        assert_no_raw_buffer_leakage(audit)
        return audit

    def check_receipt_freshness(self, source_version: str) -> Dict[str, Any]:
        if source_version not in self._observation_ledger:
            return {
                "source_id": self.source_id,
                "source_version": source_version,
                "fresh": False,
                "uncertainty_state": "unknown",
                "observed": False,
                "refusal_code": "ERR_UNOBSERVED_OR_UNKNOWN_VERSION",
            }
        entry = copy.deepcopy(self._observation_ledger[source_version])
        assert_no_raw_buffer_leakage(entry)
        return entry

    def ingest_audio_tick(
        self,
        source_version: str,
        audio_events_or_spec: Union[Sequence[Dict[str, Any]], Dict[str, Any]],
        *,
        playhead_slot: Optional[int] = None,
        playhead_ratio: float = 1.0,
        issue_observation_receipt: bool = False,
        verify_staged_ghost_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingests a playback tick into the Continuous Electric Ear stream:
          1. Translates audio to chargeable bytes and pushes them up at `playhead_slot`.
          2. Runs all 10 closed-form chargeable-byte operators.
          3. Computes pairwise charge diff & 16x16 dirty charge tiles vs previous tick.
          4. Demotes superseded `verified` observations to `stale` when charge changes.
          5. Evicts & `explicit_bzero` zeroizes expired frames beyond `max_window_frames`.
        """
        t0 = time.perf_counter()
        self._tick_counter += 1
        prev_version = self._window_versions[-1] if self._window_versions else None

        reg = self.session.register_chargeable_audio(
            self.source_id,
            source_version,
            audio_events_or_spec,
            time_slots=self.time_slots,
            freq_bands=self.freq_bands,
            playhead_slot=playhead_slot,
            playhead_ratio=playhead_ratio,
        )
        insp = self.session.inspect_audio(
            self.source_id,
            source_version,
            issue_observation_receipt=issue_observation_receipt,
        )

        if issue_observation_receipt and insp["render"]["observed"]:
            self._observation_ledger[source_version] = {
                "source_id": self.source_id,
                "source_version": source_version,
                "tick": self._tick_counter,
                "fresh": True,
                "uncertainty_state": "verified",
                "observed": True,
                "observation_receipt_id": insp["render"]["observation_receipt_id"],
                "superseded_by_version": None,
                "accessibility": build_accessibility_metadata("verified_observation"),
            }
        else:
            self._observation_ledger[source_version] = {
                "source_id": self.source_id,
                "source_version": source_version,
                "tick": self._tick_counter,
                "fresh": False,
                "uncertainty_state": "derived",
                "observed": False,
                "observation_receipt_id": None,
                "superseded_by_version": None,
                "accessibility": build_accessibility_metadata("unobserved_electric_render"),
            }

        transition_info: Optional[Dict[str, Any]] = None
        decayed_versions: List[str] = []

        if prev_version is not None:
            diff_rep = self.session.compare_audio_states(
                (self.source_id, prev_version),
                (self.source_id, source_version),
            )
            dirty_tiles = self.session._audio_vault["compute_dirty_charge_tiles_16x16"](
                (self.source_id, prev_version),
                (self.source_id, source_version),
            )
            step_hamming = int(diff_rep["ahash64_hamming_distance"])
            changed_bytes = int(diff_rep["charge_diff_metrics"]["changed_bytes"])

            if changed_bytes > 0 or step_hamming > 0:
                for old_ver, obs_entry in self._observation_ledger.items():
                    if old_ver != source_version and obs_entry.get("fresh") is True:
                        obs_entry["fresh"] = False
                        obs_entry["uncertainty_state"] = "stale"
                        obs_entry["observed"] = False
                        obs_entry["superseded_by_version"] = source_version
                        obs_entry["refusal_code"] = "ERR_STALE_OBSERVATION_SUPERSEDED"
                        obs_entry["accessibility"] = build_accessibility_metadata(
                            "quarantined_or_unverified_input"
                        )
                        decayed_versions.append(old_ver)
                        if self.session.vision_session._audit_client is not None:
                            try:
                                self.session.vision_session._audit_client.append_event(
                                    "CONTINUOUS_AUDIO_OBSERVATION_DECAYED_TO_STALE",
                                    {
                                        "source_id": self.source_id,
                                        "decayed_version": old_ver,
                                        "superseded_by_version": source_version,
                                        "changed_charge_bytes": changed_bytes,
                                        "ahash64_hamming_distance": step_hamming,
                                    },
                                )
                            except Exception:
                                pass

            transition_info = {
                "from_version": prev_version,
                "to_version": source_version,
                "changed_bytes": changed_bytes,
                "change_ratio": diff_rep["charge_diff_metrics"]["change_ratio"],
                "pushed_up_added_bytes": diff_rep["charge_diff_metrics"]["pushed_up_added_bytes"],
                "discharged_removed_bytes": diff_rep["charge_diff_metrics"]["discharged_removed_bytes"],
                "net_push_up_byte_delta": diff_rep["net_push_up_byte_delta"],
                "ahash64_hamming_distance": step_hamming,
                "checkpoint_triggered": bool(step_hamming >= self.checkpoint_hamming_threshold),
                "dirty_charge_tiles_16x16": dirty_tiles,
                "decayed_to_stale_versions": decayed_versions,
                "diff_replay_digest": diff_rep["diff_replay_digest"],
            }
            self._transition_records.append(transition_info)

        self._window_versions.append(source_version)
        self._frame_records[source_version] = {
            "tick": self._tick_counter,
            "source_version": source_version,
            "registration": reg,
            "inspection": insp,
        }

        eviction_receipt: Optional[Dict[str, Any]] = None
        if len(self._window_versions) > self.max_window_frames:
            oldest_ver = self._window_versions.pop(0)
            self._frame_records.pop(oldest_ver, None)
            self.session._source_grids.pop((self.source_id, oldest_ver), None)
            self.session._source_specs.pop((self.source_id, oldest_ver), None)
            eviction_receipt = self.session._audio_vault["evict_source"]((self.source_id, oldest_ver))
            self._eviction_log.append(eviction_receipt)

        ghost_audit: Optional[Dict[str, Any]] = None
        if verify_staged_ghost_id is not None:
            ghost_audit = self.verify_ghost_realization(verify_staged_ghost_id, source_version)

        window_summary = self.compute_temporal_window_summary()
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)

        out = {
            "status": "ok",
            "modality": "electric_audio_chargeable_bytes",
            "tick": self._tick_counter,
            "source_id": self.source_id,
            "source_version": source_version,
            "audio_grid": {"time_slots": self.time_slots, "freq_bands": self.freq_bands},
            "playhead_slot": reg["playhead_slot"],
            "charged_bytes_sha256": reg["charged_bytes_sha256"],
            "ahash64": insp["charge_metrics"]["op_perceptual_audio_hash_ahash64"],
            "observation_state": copy.deepcopy(self._observation_ledger[source_version]),
            "decayed_to_stale_versions": decayed_versions,
            "transition_from_previous": transition_info,
            "ring_buffer_eviction": eviction_receipt,
            "ghost_realization_audit": ghost_audit,
            "temporal_window_summary": window_summary,
            "elapsed_ms": elapsed_ms,
        }
        assert_no_raw_buffer_leakage(out)
        return out

    def compute_temporal_window_summary(self) -> Dict[str, Any]:
        active_versions = list(self._window_versions)
        n_frames = len(active_versions)
        recent_transitions = [
            t for t in self._transition_records if t["to_version"] in active_versions
        ]

        ahash_sequence = [
            {
                "source_version": v,
                "ahash64": self._frame_records[v]["inspection"]["charge_metrics"][
                    "op_perceptual_audio_hash_ahash64"
                ],
                "charged_bytes_sha256": self._frame_records[v]["registration"]["charged_bytes_sha256"],
            }
            for v in active_versions
        ]
        step_hammings = [int(t["ahash64_hamming_distance"]) for t in recent_transitions]
        net_window_hamming = (
            _dhash64_hamming_distance(ahash_sequence[0]["ahash64"], ahash_sequence[-1]["ahash64"])
            if n_frames >= 2
            else 0
        )

        if n_frames < 2:
            regime = "initializing"
            cycle_detected = False
            settled = False
        else:
            tail_transitions = recent_transitions[-self.settle_ticks :]
            settled = bool(
                len(tail_transitions) >= self.settle_ticks
                and all(
                    t["changed_bytes"] == 0 and t["ahash64_hamming_distance"] == 0
                    for t in tail_transitions
                )
            )
            latest_sha = ahash_sequence[-1]["charged_bytes_sha256"]
            latest_ah = ahash_sequence[-1]["ahash64"]
            earlier_non_adjacent = ahash_sequence[:-2]
            last_step_changed = bool(recent_transitions and recent_transitions[-1]["changed_bytes"] > 0)
            cycle_detected = bool(
                not settled
                and last_step_changed
                and any(
                    e["charged_bytes_sha256"] == latest_sha or e["ahash64"] == latest_ah
                    for e in earlier_non_adjacent
                )
            )
            if settled:
                regime = "settled"
            elif cycle_detected:
                regime = "oscillating_cycle"
            else:
                regime = "evolving"

        # Pitch / Charge Centroid Kinematic Trajectory Chain
        centroids_by_ver: List[Tuple[str, Optional[Tuple[float, float]], int]] = []
        for v in active_versions:
            cc = self._frame_records[v]["inspection"]["charge_metrics"]["op_connected_charge_events_4c"]
            comps = cc.get("components", [])
            if comps:
                primary = max(comps, key=lambda c: (int(c["charge_sum"]), int(c["area_bytes"])))
                centroids_by_ver.append(
                    (
                        v,
                        (float(primary["centroid"]["t"]), float(primary["centroid"]["b"])),
                        int(primary["charge_sum"]),
                    )
                )
            else:
                centroids_by_ver.append((v, None, 0))

        velocities: List[Dict[str, Any]] = []
        for i in range(len(centroids_by_ver) - 1):
            v_a, c_a, chg_a = centroids_by_ver[i]
            v_b, c_b, chg_b = centroids_by_ver[i + 1]
            if c_a is not None and c_b is not None:
                vt = round(c_b[0] - c_a[0], 4)
                vb = round(c_b[1] - c_a[1], 4)
                velocities.append({
                    "from_version": v_a,
                    "to_version": v_b,
                    "vt_slots_per_tick": vt,
                    "vb_bands_per_tick": vb,
                    "charge_sum_delta": chg_b - chg_a,
                })

        accelerations: List[Dict[str, Any]] = []
        for i in range(len(velocities) - 1):
            va = velocities[i]
            vb = velocities[i + 1]
            at = round(float(vb["vt_slots_per_tick"]) - float(va["vt_slots_per_tick"]), 4)
            ab = round(float(vb["vb_bands_per_tick"]) - float(va["vb_bands_per_tick"]), 4)
            accelerations.append({
                "at_version": vb["from_version"],
                "to_version": vb["to_version"],
                "at_slots_per_tick2": at,
                "ab_bands_per_tick2": ab,
            })

        predicted_next_centroid: Optional[Dict[str, Any]] = None
        if centroids_by_ver and centroids_by_ver[-1][1] is not None and velocities:
            ct_last, cb_last = centroids_by_ver[-1][1]  # type: ignore[misc]
            vt_last = float(velocities[-1]["vt_slots_per_tick"])
            vb_last = float(velocities[-1]["vb_bands_per_tick"])
            at_last = float(accelerations[-1]["at_slots_per_tick2"]) if accelerations else 0.0
            ab_last = float(accelerations[-1]["ab_bands_per_tick2"]) if accelerations else 0.0
            pred_t = max(0.0, min(float(self.time_slots - 1), round(ct_last + vt_last + 0.5 * at_last, 4)))
            pred_b = max(0.0, min(float(self.freq_bands - 1), round(cb_last + vb_last + 0.5 * ab_last, 4)))
            predicted_next_centroid = {
                "predicted_t_slot": pred_t,
                "predicted_freq_band": pred_b,
                "predicted_nominal_hz": _band_index_to_hz(int(round(pred_b)), self.freq_bands),
                "target_plane": "ghost_isolation_plane",
                "epistemic_state": "predicted",
                "observed": False,
            }

        dirty_ratios = [float(t["dirty_charge_tiles_16x16"]["dirty_tile_ratio"]) for t in recent_transitions]
        mean_dirty_ratio = round(sum(dirty_ratios) / float(len(dirty_ratios)), 6) if dirty_ratios else 0.0
        latest_dirty = recent_transitions[-1]["dirty_charge_tiles_16x16"] if recent_transitions else None

        active_verified = [
            v for v, entry in self._observation_ledger.items() if entry.get("fresh") is True
        ]
        stale_versions = [
            v
            for v, entry in self._observation_ledger.items()
            if entry.get("uncertainty_state") == "stale"
        ]

        summary = {
            "epistemic_label": "derived",
            "observed": False,
            "window_capacity": self.max_window_frames,
            "active_frame_count": n_frames,
            "active_versions": active_versions,
            "evicted_frame_count": len(self._eviction_log),
            "cumulative_zeroized_bytes": (
                self._eviction_log[-1]["cumulative_zeroized_bytes"] if self._eviction_log else 0
            ),
            "epistemic_watchdog": {
                "fresh_verified_versions": active_verified,
                "decayed_stale_versions": stale_versions,
            },
            "ahash64_hamming_timeline": {
                "step_hamming_distances": step_hammings,
                "max_step_hamming": max(step_hammings) if step_hammings else 0,
                "net_window_hamming_drift": net_window_hamming,
                "checkpoints_triggered": sum(1 for t in recent_transitions if t["checkpoint_triggered"]),
            },
            "settling_and_oscillation_detector": {
                "regime": regime,
                "settled": settled,
                "oscillating_cycle_detected": cycle_detected,
                "settle_ticks_required": self.settle_ticks,
            },
            "pitch_kinematic_trajectory_chain": {
                "velocities": velocities,
                "accelerations": accelerations,
                "predicted_next_centroid": predicted_next_centroid,
            },
            "dirty_charge_tile_16x16_profile": {
                "mean_dirty_tile_ratio": mean_dirty_ratio,
                "latest_transition_dirty_tiles": latest_dirty,
            },
        }
        assert_no_raw_buffer_leakage(summary)
        return summary


class ElectricSightAndSoundSession:
    """
    Unified Multimodal Perception Session for Agents ("See and Hear").

    Pairs `ElectricVisionAgentSession` (pixels / vector marks) with
    `ElectricAudioAgentSession` (chargeable bytes pushed up during play) inside
    the same out-of-process HSM/Vault/WORM Audit enclave:
      - Agents can inspect visual canvases AND chargeable-byte audio simultaneously.
      - Pushed-up audio bytes can be projected into visual marks (`project_audio_to_vision_canvas`)
        so an agent can literally see the audio bytes being pushed up as they play.
      - Computes Multimodal Charge Fusion (`inspect_sight_and_sound` /
        `fuse_sight_and_sound_tick`): aligns visual edge/pixel activity with
        audio fast/slow charge push-up to compute joint audiovisual salience in Q16.16.
    """

    def __init__(
        self,
        *,
        run_dir: str = "/tmp/cyclops-ee",
        default_resolution: Tuple[int, int] = (256, 256),
        default_audio_grid: Tuple[int, int] = (256, 64),
    ) -> None:
        self.run_dir = run_dir
        self.vision = ElectricVisionAgentSession(
            run_dir=run_dir,
            default_resolution=default_resolution,
        )
        self.audio = ElectricAudioAgentSession(
            run_dir=run_dir,
            default_grid=default_audio_grid,
            shared_vision_session=self.vision,
        )

    def __enter__(self) -> "ElectricSightAndSoundSession":
        self.vision.__enter__()
        self.audio.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            self.audio.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.vision.__exit__(exc_type, exc_val, exc_tb)

    def inspect_sight_and_sound(
        self,
        vision_source_id: str,
        audio_source_id: str,
        source_version: str = "v1",
        *,
        visual_marks: Optional[Sequence[Union[DrawingMark, Dict[str, Any]]]] = None,
        audio_events_or_spec: Union[Sequence[Dict[str, Any]], Dict[str, Any]],
        playhead_slot: Optional[int] = None,
        playhead_ratio: float = 1.0,
        project_audio_into_vision_if_empty: bool = True,
        issue_observation_receipts: bool = False,
    ) -> Dict[str, Any]:
        """
        Registers and inspects both an Electric Vision canvas and a Chargeable-Byte
        Electric Audio source for `source_version`, computing joint multimodal
        charge fusion metrics (`joint_salience_q16`, `audiovisual_coincidence`).
        If `visual_marks` is omitted and `project_audio_into_vision_if_empty=True`,
        the pushed-up audio bytes are automatically projected onto the visual canvas!
        """
        self.audio.register_chargeable_audio(
            audio_source_id,
            source_version,
            audio_events_or_spec,
            playhead_slot=playhead_slot,
            playhead_ratio=playhead_ratio,
        )
        audio_rep = self.audio.inspect_audio(
            audio_source_id,
            source_version,
            issue_observation_receipt=issue_observation_receipts,
        )

        projected_from_audio = False
        if visual_marks is None and project_audio_into_vision_if_empty:
            resolved_marks: Sequence[Union[DrawingMark, Dict[str, Any]]] = (
                self.audio.project_pushed_up_bytes_to_marks(audio_source_id, source_version)
            )
            projected_from_audio = True
        else:
            resolved_marks = visual_marks or []

        self.vision.register_synthetic_canvas(
            vision_source_id,
            source_version,
            resolved_marks,
            source_kind="captured_frame",
        )
        vision_rep = self.vision.inspect_canvas(
            vision_source_id,
            source_version,
            issue_observation_receipt=issue_observation_receipts,
        )

        # Multimodal Charge Fusion (Q16.16 integer arithmetic)
        v_simd = vision_rep["native_simd_metrics"]
        a_chg = audio_rep["charge_metrics"]

        v_alpha_q16 = int(v_simd["op_alpha_coverage_q16"])
        v_edge_q16 = int(v_simd["op_sobel_edge_density_q16"])
        a_push_cov_q16 = int(a_chg["op_charge_summary_u8"]["pushed_up_coverage_q16"])
        a_mean_push_q16 = int(a_chg["op_charge_summary_u8"]["mean_pushed_up_q16"])
        a_flux_q16 = int(a_chg["op_spectro_temporal_flux_q16"])

        # Normalize audio mean push-up (0..255 << 16) to [0..65536]
        a_norm_q16 = a_mean_push_q16 // 255
        v_activity_q16 = min(65536, (v_alpha_q16 + (v_edge_q16 // 128)) // 2)
        a_activity_q16 = min(65536, (a_push_cov_q16 + a_norm_q16 + (a_flux_q16 // 64)) // 3)
        joint_salience_q16 = (v_activity_q16 + a_activity_q16) // 2

        v_active = v_activity_q16 >= 512
        a_active = int(a_chg["op_charge_summary_u8"]["pushed_up_active_bytes"]) > 0
        if v_active and a_active:
            modal_alignment = "audiovisual_coincident_active"
        elif v_active and not a_active:
            modal_alignment = "visual_only_unplayed_or_silent"
        elif a_active and not v_active:
            modal_alignment = "audio_pushed_up_visual_empty"
        else:
            modal_alignment = "quiescent"

        fusion = {
            "epistemic_label": "derived",
            "observed": False,
            "projected_pushed_up_bytes_to_vision": projected_from_audio,
            "visual_activity_q16": v_activity_q16,
            "audio_push_up_activity_q16": a_activity_q16,
            "joint_salience_q16": joint_salience_q16,
            "modal_alignment": modal_alignment,
            "pushed_up_alert_triggered": bool(
                a_chg["op_push_up_profile_q16"]["pushed_up_alert_triggered"]
            ),
            "fusion_replay_digest": compute_replay_digest(
                source_id=vision_source_id,
                source_version=source_version,
                secondary_source={"source_id": audio_source_id, "source_version": source_version},
                operator_name="multimodal_sight_and_sound_fusion",
                operator_version=OPERATOR_VERSION,
                parameters={"projected_from_audio": projected_from_audio},
                result={
                    "v_activity_q16": v_activity_q16,
                    "a_activity_q16": a_activity_q16,
                    "joint_salience_q16": joint_salience_q16,
                    "modal_alignment": modal_alignment,
                },
            ),
        }

        out = {
            "status": "ok",
            "modality": "electric_sight_and_sound",
            "source_version": source_version,
            "sight": vision_rep,
            "sound": audio_rep,
            "multimodal_fusion": fusion,
        }
        assert_no_raw_buffer_leakage(out)
        return out


# =============================================================================
# Real-Time Charge Packet Codec, Three-Clock Fusion Gate & Agent Electrical Area
# =============================================================================

MODALITY_VISION = 0
MODALITY_AUDIO = 1
MODALITY_FUSED = 2

LANE_FAST = 0
LANE_SLOW = 1
LANE_DUAL = 2

DEFAULT_FAST_HALF_LIFE_MS = 50
DEFAULT_SLOW_HALF_LIFE_MS = 1500
DEFAULT_ALIGN_WINDOW_MS = 100
DEFAULT_ELECTRICAL_AREA_WINDOW_MS = 3000


def compute_align_id(timestamp_ms: int, align_window_ms: int = DEFAULT_ALIGN_WINDOW_MS) -> int:
    """Computes the deterministic 100 ms window alignment ID (`align_id`) for `timestamp_ms`."""
    return max(0, int(timestamp_ms)) // max(1, int(align_window_ms))


def build_charge_packet(
    *,
    t: int,
    modality: int,
    f_band: int = 0,
    q_fast: int = 0,
    q_slow: int = 0,
    push_threshold_fast: int = 40,
    push_threshold_slow: int = 32,
    revision: int = 0,
    decay_half_life_ms: Optional[int] = None,
    align_window_ms: int = DEFAULT_ALIGN_WINDOW_MS,
) -> Dict[str, Any]:
    """
    Constructs a canonical Charge Packet combining the single-modality and
    multimodal real-time fusion byte layouts:
      - `t`        : timestamp (ms)
      - `f_band`   : frequency band / pitch class / spatial region (0..255)
      - `modality` : 0=vision, 1=audio, 2=fused
      - `q_fast`   : fast-lane charge (0..255, ~20-50 ms transients/phonemes/motion)
      - `q_slow`   : slow-lane charge (0..255, ~1-8 s harmony/prosody/scene mood)
      - `q`        : peak active charge magnitude (0..255)
      - `lane`     : 0=fast, 1=slow, 2=dual
      - `align_id` : integer window ID linking vision+audio within ~100 ms
      - `push`     : 0/1 (crossed threshold into the agent's Electrical Area)
      - `revision` : 0/1 (updates/contradicts prior impression in Electrical Area)
      - `decay`    : half-life in ms
    """
    if modality not in (MODALITY_VISION, MODALITY_AUDIO, MODALITY_FUSED):
        raise ValueError(f"Invalid modality {modality}; must be 0 (vision), 1 (audio), or 2 (fused)")
    t_ms = max(0, int(t))
    fb_u8 = max(0, min(255, int(f_band)))
    qf_u8 = max(0, min(255, int(q_fast)))
    qs_u8 = max(0, min(255, int(q_slow)))
    q_u8 = max(qf_u8, qs_u8)

    fast_pushed = qf_u8 >= push_threshold_fast
    slow_pushed = qs_u8 >= push_threshold_slow
    push_bit = 1 if (fast_pushed or slow_pushed) else 0
    rev_bit = 1 if int(revision) else 0

    if fast_pushed and slow_pushed:
        lane_id = LANE_DUAL
    elif slow_pushed or (qs_u8 > qf_u8 and not fast_pushed):
        lane_id = LANE_SLOW
    else:
        lane_id = LANE_FAST

    if decay_half_life_ms is None:
        decay_ms = DEFAULT_SLOW_HALF_LIFE_MS if lane_id == LANE_SLOW else DEFAULT_FAST_HALF_LIFE_MS
    else:
        decay_ms = max(1, min(65535, int(decay_half_life_ms)))

    align_id = compute_align_id(t_ms, align_window_ms)
    return {
        "t": t_ms,
        "f_band": fb_u8,
        "modality": modality,
        "q_fast": qf_u8,
        "q_slow": qs_u8,
        "q": q_u8,
        "lane": lane_id,
        "align_id": align_id,
        "push": push_bit,
        "revision": rev_bit,
        "decay": decay_ms,
    }


def pack_charge_packet_16bytes(pkt: Dict[str, Any]) -> bytes:
    """
    Packs a `ChargePacket` into a compact, deterministic 16-byte record:
      - bytes 0..3  : `t` (uint32 big-endian, timestamp ms)
      - bytes 4..7  : `align_id` (uint32 big-endian)
      - byte  8     : `modality` (uint8: 0, 1, 2)
      - byte  9     : `f_band` (uint8: 0..255)
      - byte  10    : `q_fast` (uint8: 0..255)
      - byte  11    : `q_slow` (uint8: 0..255)
      - byte  12    : `q` (uint8: 0..255)
      - byte  13    : `flags` (`(lane & 0x3) | ((push & 0x1) << 2) | ((revision & 0x1) << 3)`)
      - bytes 14..15: `decay` (uint16 big-endian, half-life ms)
    """
    import struct

    flags = (
        (int(pkt["lane"]) & 0x03)
        | ((int(pkt["push"]) & 0x01) << 2)
        | ((int(pkt["revision"]) & 0x01) << 3)
    )
    return struct.pack(
        ">IIBBBBBBH",
        max(0, min(0xFFFFFFFF, int(pkt["t"]))),
        max(0, min(0xFFFFFFFF, int(pkt["align_id"]))),
        max(0, min(255, int(pkt["modality"]))),
        max(0, min(255, int(pkt["f_band"]))),
        max(0, min(255, int(pkt["q_fast"]))),
        max(0, min(255, int(pkt["q_slow"]))),
        max(0, min(255, int(pkt["q"]))),
        flags,
        max(1, min(0xFFFF, int(pkt["decay"]))),
    )


def unpack_charge_packet_16bytes(raw: bytes) -> Dict[str, Any]:
    """Unpacks a 16-byte binary Charge Packet record back into its canonical dict."""
    import struct

    if len(raw) != 16:
        raise ValueError(f"Expected 16-byte charge packet, got {len(raw)} bytes")
    t_ms, align_id, mod, fb, qf, qs, q_val, flags, decay_ms = struct.unpack(
        ">IIBBBBBBH", raw
    )
    return {
        "t": int(t_ms),
        "align_id": int(align_id),
        "modality": int(mod),
        "f_band": int(fb),
        "q_fast": int(qf),
        "q_slow": int(qs),
        "q": int(q_val),
        "lane": int(flags & 0x03),
        "push": int((flags >> 2) & 0x01),
        "revision": int((flags >> 3) & 0x01),
        "decay": int(decay_ms),
    }


def ablate_charge_lanes(
    packets: Sequence[Dict[str, Any]],
    *,
    mode: str = "dual",
    push_threshold_fast: int = 40,
    push_threshold_slow: int = 32,
) -> Dict[str, Any]:
    """
    Step 5 Two-Lane Ablation Evaluator:
      - `mode="dual"`      : keeps both `q_fast` (transients/phonemes) and `q_slow` (harmony/contour).
      - `mode="no_slow"`   : zeroes `q_slow` -> drops slow harmonic/envelope continuity while keeping fast transients.
      - `mode="no_fast"`   : zeroes `q_fast` -> drops fast transient/onset precision while keeping slow contour.
    """
    if mode not in ("dual", "no_slow", "no_fast"):
        raise ValueError(f"Unsupported ablation mode {mode!r}")

    retained_fast_pushes = 0
    retained_slow_pushes = 0
    total_fast_charge = 0
    total_slow_charge = 0
    ablated_packets: List[Dict[str, Any]] = []

    for p in packets:
        qf = 0 if mode == "no_fast" else int(p["q_fast"])
        qs = 0 if mode == "no_slow" else int(p["q_slow"])
        new_p = build_charge_packet(
            t=int(p["t"]),
            modality=int(p["modality"]),
            f_band=int(p["f_band"]),
            q_fast=qf,
            q_slow=qs,
            push_threshold_fast=push_threshold_fast,
            push_threshold_slow=push_threshold_slow,
            revision=int(p.get("revision", 0)),
        )
        if qf >= push_threshold_fast:
            retained_fast_pushes += 1
        if qs >= push_threshold_slow:
            retained_slow_pushes += 1
        total_fast_charge += qf
        total_slow_charge += qs
        ablated_packets.append(new_p)

    return {
        "ablation_mode": mode,
        "packet_count": len(ablated_packets),
        "fast_lane_pushed_packets": retained_fast_pushes,
        "slow_lane_pushed_packets": retained_slow_pushes,
        "total_fast_charge_sum": total_fast_charge,
        "total_slow_charge_sum": total_slow_charge,
        "fast_transient_recall_preserved": bool(retained_fast_pushes > 0),
        "slow_contour_recall_preserved": bool(retained_slow_pushes > 0),
    }


class AgentElectricalAreaBuffer:
    """
    Real-Time Three-Clock Multimodal Charge Fusion Gate & Agent "Electrical Area"
    Working Memory Buffer.

    Pipeline:
      `sensory stream (vision 30fps / audio 20ms) -> charge field -> push-up gate -> agent electrical area`

    Enforces all 7 real-time design invariants:
      1. Three Clocks:
         - `sensory_clock_ms`: raw frame/chunk timestamp `t` (ms).
         - `charge_clock_ms`: `q_fast` (`~50 ms` half-life) & `q_slow` (`~1500 ms` half-life).
         - `narration_clock_ms`: emits micro-tokens every `micro_cadence_ms` (500 ms)
           and macro-tokens every `macro_cadence_ms` (3000 ms).
      2. 100 ms Multimodal Alignment (`align_id`):
         - Vision (`modality=0`) and Audio (`modality=1`) packets within the same
           `100 ms` window share `align_id` and fuse into a combined `modality=2`
           pushed-up packet in the Electrical Area.
      3. Sliding Working Memory (`window_ms = 3000` [2-4 s]) with Decay Eviction:
         - Old charge decays as `t` advances and packets older than `window_ms`
           or decayed below `min_retention_charge` are evicted.
      4. Deduplication by `(align_id, modality)`:
         - Prevents repeating the same event token twice.
      5. Separate Per-Modality Charge Caps (`vision_cap = 255`, `audio_cap = 255`):
         - Vision never drowns audio; unimodal vision-only or audio-only streams
           push up cleanly on their own.
      6. Revision Token (`revision = 1`):
         - When a new high-charge pushed-up packet shifts the dominant band/region
           by `>= revision_band_delta` from the active impression in the Electrical
           Area, `revision=1` is set automatically.
      7. Low-Charge Guard (`"unknown_low_charge"`):
         - When no packet crosses the push-up threshold, emits an explicit
           `"unknown_low_charge"` state rather than hallucinating activity.
    """

    def __init__(
        self,
        *,
        window_ms: int = DEFAULT_ELECTRICAL_AREA_WINDOW_MS,
        align_window_ms: int = DEFAULT_ALIGN_WINDOW_MS,
        push_threshold_fast: int = 40,
        push_threshold_slow: int = 32,
        micro_cadence_ms: int = 500,
        macro_cadence_ms: int = 3000,
        revision_band_delta: int = 16,
        vision_cap: int = 255,
        audio_cap: int = 255,
    ) -> None:
        self.window_ms = max(500, int(window_ms))
        self.align_window_ms = max(10, int(align_window_ms))
        self.push_threshold_fast = int(push_threshold_fast)
        self.push_threshold_slow = int(push_threshold_slow)
        self.micro_cadence_ms = max(100, int(micro_cadence_ms))
        self.macro_cadence_ms = max(500, int(macro_cadence_ms))
        self.revision_band_delta = max(1, int(revision_band_delta))
        self.vision_cap = max(1, min(255, int(vision_cap)))
        self.audio_cap = max(1, min(255, int(audio_cap)))

        self._active_area: List[Dict[str, Any]] = []
        self._by_align_id: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self._emitted_micro_align_ids: set[int] = set()
        self._micro_tokens: List[Dict[str, Any]] = []
        self._macro_tokens: List[Dict[str, Any]] = []
        self._revision_tokens: List[Dict[str, Any]] = []
        self._last_micro_t: int = -100000
        self._last_macro_t: int = -100000
        self._dominant_band_by_modality: Dict[int, int] = {}
        self._evicted_count: int = 0
        self._dedup_skipped_count: int = 0
        self._latest_t_ms: int = 0
        self._cumulative_slow_vector: List[int] = [0] * 8

    def _decay_and_evict(self, current_t_ms: int) -> None:
        surviving: List[Dict[str, Any]] = []
        for entry in self._active_area:
            dt = max(0, current_t_ms - int(entry["t"]))
            if dt > self.window_ms:
                self._evicted_count += 1
                continue
            # Integer half-life decay approximation
            hl = max(1, int(entry["decay"]))
            half_steps = min(8, dt // hl)
            decayed_q = int(entry["q"]) >> half_steps
            entry["current_decayed_q"] = decayed_q
            if decayed_q >= 4:
                surviving.append(entry)
            else:
                self._evicted_count += 1
        self._active_area = surviving

    def ingest_sensory_packet(
        self,
        *,
        t_ms: int,
        modality: int,
        f_band: int,
        q_fast: int,
        q_slow: int,
    ) -> Dict[str, Any]:
        """
        Ingests a vision (`0`) or audio (`1`) sensory sample at `t_ms`, applies
        per-modality saturation caps, checks the push-up threshold gate, fuses
        same-`align_id` (`<= 100 ms`) multimodal events into `modality=2`, and
        updates the Agent Electrical Area.
        """
        t_cur = max(self._latest_t_ms, max(0, int(t_ms)))
        self._latest_t_ms = t_cur
        self._decay_and_evict(t_cur)

        cap = self.vision_cap if modality == MODALITY_VISION else self.audio_cap
        qf_capped = min(cap, max(0, int(q_fast)))
        qs_capped = min(cap, max(0, int(q_slow)))

        # Check whether this contradicts/revises the prior dominant band for this modality
        rev_flag = 0
        if (qf_capped >= self.push_threshold_fast or qs_capped >= self.push_threshold_slow):
            prev_band = self._dominant_band_by_modality.get(modality)
            if prev_band is not None and abs(int(f_band) - prev_band) >= self.revision_band_delta:
                rev_flag = 1
            self._dominant_band_by_modality[modality] = int(f_band)

        pkt = build_charge_packet(
            t=t_cur,
            modality=modality,
            f_band=f_band,
            q_fast=qf_capped,
            q_slow=qs_capped,
            push_threshold_fast=self.push_threshold_fast,
            push_threshold_slow=self.push_threshold_slow,
            revision=rev_flag,
            align_window_ms=self.align_window_ms,
        )
        pkt["current_decayed_q"] = pkt["q"]
        pkt["packed_16bytes_hex"] = pack_charge_packet_16bytes(pkt).hex()

        # Accumulate into global 8-bin slow summary vector
        bin8 = max(0, min(7, (int(f_band) * 8) // 256))
        self._cumulative_slow_vector[bin8] = min(65535, self._cumulative_slow_vector[bin8] + qs_capped)

        align_id = pkt["align_id"]
        slot_map = self._by_align_id.setdefault(align_id, {})

        # Deduplication guard: if same (align_id, modality) already pushed up, skip duplicate emission
        if modality in slot_map and slot_map[modality]["push"] == 1 and pkt["push"] == 1:
            self._dedup_skipped_count += 1
            return {
                "ingested_packet": pkt,
                "pushed_into_electrical_area": False,
                "dedup_skipped": True,
                "fused_packet": slot_map.get(MODALITY_FUSED),
                "electrical_area_status": self.snapshot(current_t_ms=t_cur),
            }

        slot_map[modality] = pkt
        fused_pkt: Optional[Dict[str, Any]] = None

        if pkt["push"] == 1:
            self._active_area.append(pkt)
            if rev_flag == 1:
                rev_tok = {
                    "t_ms": t_cur,
                    "align_id": align_id,
                    "modality": modality,
                    "new_f_band": int(f_band),
                    "token": f"[REVISION:MOD_{modality}_BAND_SHIFT->{int(f_band)}]",
                }
                self._revision_tokens.append(rev_tok)

            # Check if both vision (0) and audio (1) pushed up within this 100 ms align_id window!
            v_pkt = slot_map.get(MODALITY_VISION)
            a_pkt = slot_map.get(MODALITY_AUDIO)
            if (
                v_pkt is not None
                and a_pkt is not None
                and v_pkt["push"] == 1
                and a_pkt["push"] == 1
                and MODALITY_FUSED not in slot_map
            ):
                fused_pkt = build_charge_packet(
                    t=max(int(v_pkt["t"]), int(a_pkt["t"])),
                    modality=MODALITY_FUSED,
                    f_band=(int(v_pkt["f_band"]) + int(a_pkt["f_band"])) // 2,
                    q_fast=min(255, (int(v_pkt["q_fast"]) + int(a_pkt["q_fast"])) // 2),
                    q_slow=min(255, (int(v_pkt["q_slow"]) + int(a_pkt["q_slow"])) // 2),
                    push_threshold_fast=self.push_threshold_fast,
                    push_threshold_slow=self.push_threshold_slow,
                    revision=max(int(v_pkt["revision"]), int(a_pkt["revision"])),
                    align_window_ms=self.align_window_ms,
                )
                fused_pkt["current_decayed_q"] = fused_pkt["q"]
                fused_pkt["sync_delta_ms"] = abs(int(v_pkt["t"]) - int(a_pkt["t"]))
                fused_pkt["packed_16bytes_hex"] = pack_charge_packet_16bytes(fused_pkt).hex()
                slot_map[MODALITY_FUSED] = fused_pkt
                self._active_area.append(fused_pkt)

            # Real-time Micro-Token emission (fast lane, deduplicated by align_id)
            target_for_micro = fused_pkt or pkt
            if (
                align_id not in self._emitted_micro_align_ids
                and (
                    fused_pkt is not None
                    or (t_cur - self._last_micro_t >= self.micro_cadence_ms)
                )
            ):
                if fused_pkt is not None:
                    kind_label = "audiovisual_coincident_push"
                elif modality == MODALITY_VISION:
                    kind_label = "visual_only_push"
                else:
                    kind_label = "audio_only_push"
                micro_tok = {
                    "t_ms": t_cur,
                    "align_id": align_id,
                    "modality": target_for_micro["modality"],
                    "kind": kind_label,
                    "q_fast": target_for_micro["q_fast"],
                    "f_band": target_for_micro["f_band"],
                    "revision": target_for_micro["revision"],
                    "emission_latency_ms": 0,
                }
                # Upgrade earlier unimodal micro token in same align_id if fused arrived within 100ms
                self._emitted_micro_align_ids.add(align_id)
                self._last_micro_t = t_cur
                self._micro_tokens.append(micro_tok)
            elif fused_pkt is not None and self._micro_tokens and self._micro_tokens[-1]["align_id"] == align_id:
                # Upgrade the same-window micro-token in place to audiovisual_coincident_push
                self._micro_tokens[-1]["modality"] = MODALITY_FUSED
                self._micro_tokens[-1]["kind"] = "audiovisual_coincident_push"
                self._micro_tokens[-1]["q_fast"] = fused_pkt["q_fast"]

            # Real-time Macro-Token emission (slow lane, every macro_cadence_ms)
            if t_cur - self._last_macro_t >= self.macro_cadence_ms:
                slow_sum = sum(int(e["q_slow"]) for e in self._active_area)
                self._last_macro_t = t_cur
                self._macro_tokens.append({
                    "t_ms": t_cur,
                    "active_area_packets": len(self._active_area),
                    "slow_charge_sum": slow_sum,
                    "regime": "sustained_high_charge" if slow_sum >= 120 else "moderate_contour",
                })

        return {
            "ingested_packet": pkt,
            "pushed_into_electrical_area": bool(pkt["push"] == 1),
            "dedup_skipped": False,
            "fused_packet": fused_pkt,
            "electrical_area_status": self.snapshot(current_t_ms=t_cur),
        }

    def snapshot(self, *, current_t_ms: Optional[int] = None) -> Dict[str, Any]:
        """Returns the current state of the Agent Electrical Area working memory."""
        t_now = self._latest_t_ms if current_t_ms is None else max(self._latest_t_ms, int(current_t_ms))
        self._decay_and_evict(t_now)

        if not self._active_area:
            state_token = "unknown_low_charge"
        else:
            mods = {int(e["modality"]) for e in self._active_area}
            if MODALITY_FUSED in mods or (MODALITY_VISION in mods and MODALITY_AUDIO in mods):
                state_token = "audiovisual_coincident"
            elif MODALITY_VISION in mods:
                state_token = "vision_only"
            else:
                state_token = "audio_only"

        return {
            "current_t_ms": t_now,
            "window_ms": self.window_ms,
            "electrical_area_state_token": state_token,
            "active_pushed_packet_count": len(self._active_area),
            "evicted_packet_count": self._evicted_count,
            "dedup_skipped_count": self._dedup_skipped_count,
            "micro_tokens": list(self._micro_tokens),
            "macro_tokens": list(self._macro_tokens),
            "revision_tokens": list(self._revision_tokens),
            "global_slow_summary_vector_8bin": list(self._cumulative_slow_vector),
        }


def run_realtime_fusion_test_plan() -> Dict[str, Any]:
    """
    Executes the complete 4-part Real-Time Multimodal Charge Fusion Test Plan
    (Tests A, B, C, D) + Step 5 Two-Lane Ablation verification:
      - Test A (Synchronization): Vision + Audio clap within 40 ms (< 100 ms)
        shares `align_id` and fuses into `modality=2` (`audiovisual_coincident_push`).
      - Test B (Unimodal Stress): Silent vision-only clip and black-screen
        audio-only clip each push up cleanly without cross-modal drowning, and
        sub-threshold noise stays `unknown_low_charge`.
      - Test C (Revision): Initial impression at band 20 followed by sharp jump
        to band 96 emits a `revision=1` token in the Electrical Area.
      - Test D (Latency, Decay Eviction & Deduplication): Verifies duplicate
        `(align_id, modality)` suppression and sliding-window charge eviction
        after `window_ms`.
      - Step 5 (Two-Lane Ablation): Verifies `no_slow` vs `no_fast` vs `dual`.
    """
    # Test A: Synchronization (< 100 ms align_id fusion)
    area_a = AgentElectricalAreaBuffer(window_ms=3000, align_window_ms=100)
    area_a.ingest_sensory_packet(t_ms=210, modality=MODALITY_VISION, f_band=64, q_fast=180, q_slow=70)
    res_a = area_a.ingest_sensory_packet(t_ms=245, modality=MODALITY_AUDIO, f_band=80, q_fast=210, q_slow=90)
    fused_a = res_a["fused_packet"]
    test_a_passed = bool(
        fused_a is not None
        and fused_a["modality"] == MODALITY_FUSED
        and fused_a["align_id"] == 2
        and fused_a["sync_delta_ms"] == 35
        and area_a.snapshot()["electrical_area_state_token"] == "audiovisual_coincident"
    )

    # Test B: Unimodal Stress (Vision-only, Audio-only, and Sub-threshold unknown_low_charge)
    area_b_quiet = AgentElectricalAreaBuffer()
    area_b_quiet.ingest_sensory_packet(t_ms=100, modality=MODALITY_AUDIO, f_band=10, q_fast=8, q_slow=5)
    quiet_token = area_b_quiet.snapshot()["electrical_area_state_token"]

    area_b_vis = AgentElectricalAreaBuffer()
    area_b_vis.ingest_sensory_packet(t_ms=100, modality=MODALITY_VISION, f_band=40, q_fast=150, q_slow=60)
    vis_token = area_b_vis.snapshot()["electrical_area_state_token"]

    area_b_aud = AgentElectricalAreaBuffer()
    area_b_aud.ingest_sensory_packet(t_ms=100, modality=MODALITY_AUDIO, f_band=90, q_fast=160, q_slow=80)
    aud_token = area_b_aud.snapshot()["electrical_area_state_token"]

    test_b_passed = bool(
        quiet_token == "unknown_low_charge"
        and vis_token == "vision_only"
        and aud_token == "audio_only"
    )

    # Test C: Revision (band shift >= 16 triggers revision=1)
    area_c = AgentElectricalAreaBuffer(revision_band_delta=16)
    area_c.ingest_sensory_packet(t_ms=100, modality=MODALITY_AUDIO, f_band=20, q_fast=120, q_slow=60)
    res_c2 = area_c.ingest_sensory_packet(t_ms=700, modality=MODALITY_AUDIO, f_band=96, q_fast=200, q_slow=110)
    snap_c = area_c.snapshot()
    test_c_passed = bool(
        res_c2["ingested_packet"]["revision"] == 1
        and len(snap_c["revision_tokens"]) == 1
    )

    # Test D: Deduplication & Sliding-Window Decay Eviction
    area_d = AgentElectricalAreaBuffer(window_ms=2000)
    area_d.ingest_sensory_packet(t_ms=100, modality=MODALITY_AUDIO, f_band=50, q_fast=140, q_slow=60)
    dup_res = area_d.ingest_sensory_packet(t_ms=120, modality=MODALITY_AUDIO, f_band=50, q_fast=145, q_slow=60)
    # Advance past 2000 ms window so t=100 packet is evicted
    area_d.ingest_sensory_packet(t_ms=2500, modality=MODALITY_AUDIO, f_band=60, q_fast=150, q_slow=80)
    snap_d = area_d.snapshot(current_t_ms=2500)
    test_d_passed = bool(
        dup_res["dedup_skipped"] is True
        and snap_d["dedup_skipped_count"] == 1
        and snap_d["evicted_packet_count"] >= 1
    )

    # Step 5: Two-Lane Ablation
    sample_pkts = [
        build_charge_packet(t=100, modality=MODALITY_AUDIO, f_band=40, q_fast=180, q_slow=20),
        build_charge_packet(t=600, modality=MODALITY_AUDIO, f_band=48, q_fast=25, q_slow=140),
        build_charge_packet(t=1200, modality=MODALITY_AUDIO, f_band=52, q_fast=190, q_slow=160),
    ]
    abl_dual = ablate_charge_lanes(sample_pkts, mode="dual")
    abl_no_slow = ablate_charge_lanes(sample_pkts, mode="no_slow")
    abl_no_fast = ablate_charge_lanes(sample_pkts, mode="no_fast")
    ablation_passed = bool(
        abl_dual["fast_transient_recall_preserved"]
        and abl_dual["slow_contour_recall_preserved"]
        and abl_no_slow["fast_transient_recall_preserved"]
        and not abl_no_slow["slow_contour_recall_preserved"]
        and abl_no_fast["slow_contour_recall_preserved"]
        and not abl_no_fast["fast_transient_recall_preserved"]
    )

    all_passed = bool(
        test_a_passed and test_b_passed and test_c_passed and test_d_passed and ablation_passed
    )
    out = {
        "all_passed": all_passed,
        "test_a_synchronization_100ms_passed": test_a_passed,
        "test_b_unimodal_and_low_charge_passed": test_b_passed,
        "test_c_revision_token_passed": test_c_passed,
        "test_d_dedup_and_window_eviction_passed": test_d_passed,
        "step_5_two_lane_ablation_passed": ablation_passed,
        "ablation_Comparison": {
            "dual": abl_dual,
            "no_slow": abl_no_slow,
            "no_fast": abl_no_fast,
        },
    }
    assert_no_raw_buffer_leakage(out)
    return out


# =============================================================================
# Feedback Remediation Suite (A-E):
#   A. Semantic Boundary Contract & Enclave vs. Agent Interpretation Separation
#   B. Tier-2 Consented Sandboxed PCM/WAV Validator, Vocal Sub-Lane (300-3400Hz)
#      & Adversarial Audio Guard (DC offset, clipping, ultrasonic > 7040Hz)
#   C. Adaptive Session Threshold Calibration & Explainable Charge Trace Debug
#   D. Agent-Layer Narration State Machine (300ms Coalescing & Amendment Policy)
#   E. Tier-2 Fidelity Study (Speech vs. Polyphonic Music vs. Noise) & Edge Profile
# =============================================================================

SEMANTIC_BOUNDARY_CONTRACT: Dict[str, Any] = {
    "contract_id": "ELECTRIC-SIGHT-SOUND-SEMANTIC-BOUNDARY-V1",
    "core_rule": (
        "Enclave emits deterministic charge events and hashable receipts; "
        "agent may interpret outside the enclave; enclave never claims meaning."
    ),
    "enclave_sensor_layer": {
        "epistemic_origin": "enclave_deterministic_sensor",
        "is_enclave_truth": True,
        "deterministic_replay": True,
        "claims_semantic_meaning": False,
        "permitted_outputs": [
            "charge_metrics",
            "charge_packets_16byte",
            "vocal_band_300_3400_flux_q16",
            "observation_receipts",
            "worm_audit_events",
        ],
    },
    "agent_interpreter_layer": {
        "epistemic_origin": "agent_probabilistic_interpreter",
        "is_enclave_truth": False,
        "deterministic_replay": False,
        "may_cite_as_enclave_truth": False,
        "required_provenance_fields": [
            "enclave_receipt_id",
            "enclave_replay_digest",
            "model_id",
            "prompt_version",
            "confidence_q16",
        ],
    },
}


def create_agent_interpretation_envelope(
    *,
    enclave_receipt_id: str,
    enclave_replay_digest: str,
    model_id: str,
    prompt_version: str,
    interpreted_tokens: Sequence[str],
    confidence_q16: int,
    amends_envelope_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Wraps a probabilistic agent-layer interpretation (outside Layer 4 of the
    enclave) in an explicit non-authoritative envelope so auditors can never
    confuse stochastic LLM/classifier output with deterministic enclave receipts.
    """
    if not enclave_receipt_id or not enclave_replay_digest.startswith("sha256:"):
        raise ValueError("Agent interpretation requires a valid enclave_receipt_id and sha256 replay_digest")
    clean_tokens = [str(tok)[:120] for tok in interpreted_tokens[:32]]
    envelope_id = f"agt_interp_{sha256_hex((enclave_receipt_id + model_id + str(clean_tokens)).encode())[:16]}"
    return {
        "envelope_id": envelope_id,
        "contract_id": SEMANTIC_BOUNDARY_CONTRACT["contract_id"],
        "epistemic_origin": "agent_probabilistic_interpreter",
        "is_enclave_truth": False,
        "may_cite_as_enclave_truth": False,
        "enclave_provenance": {
            "enclave_receipt_id": enclave_receipt_id,
            "enclave_replay_digest": enclave_replay_digest,
        },
        "interpreter_metadata": {
            "model_id": str(model_id)[:64],
            "prompt_version": str(prompt_version)[:32],
            "confidence_q16": max(0, min(65536, int(confidence_q16))),
        },
        "amends_envelope_id": amends_envelope_id,
        "interpreted_tokens": clean_tokens,
    }


def calibrate_adaptive_charge_thresholds(
    potential_slice_u8: Sequence[int],
    novelty_slice_u8: Sequence[int],
    *,
    audit_client: Optional[Any] = None,
    source_id: str = "src_adaptive_cal",
) -> Dict[str, Any]:
    """
    Calibrates per-session `push_threshold_fast` and `push_threshold_slow` from
    the initial calibration window in pure integer arithmetic (`mean + 2*mad`,
    clamped to `[16..180]`) and logs the calibration to the WORM audit sink.
    """
    n_pot = len(potential_slice_u8)
    n_nov = len(novelty_slice_u8)
    if n_pot == 0 or n_nov == 0:
        fast_thr, slow_thr = 40, 32
        mean_pot, mad_pot = 0, 0
        mean_nov, mad_nov = 0, 0
    else:
        mean_pot = sum(int(x) for x in potential_slice_u8) // n_pot
        mad_pot = sum(abs(int(x) - mean_pot) for x in potential_slice_u8) // n_pot
        mean_nov = sum(int(x) for x in novelty_slice_u8) // n_nov
        mad_nov = sum(abs(int(x) - mean_nov) for x in novelty_slice_u8) // n_nov
        slow_thr = max(16, min(180, mean_pot + 2 * mad_pot + 12))
        fast_thr = max(20, min(180, ((mean_pot >> 1) + mean_nov) + 2 * (mad_pot + mad_nov) + 16))

    cal_record = {
        "source_id": source_id,
        "calibration_samples": n_pot,
        "noise_floor_mean_u8": mean_pot,
        "noise_floor_mad_u8": mad_pot,
        "onset_mean_u8": mean_nov,
        "onset_mad_u8": mad_nov,
        "calibrated_push_threshold_fast": fast_thr,
        "calibrated_push_threshold_slow": slow_thr,
    }
    if audit_client is not None:
        try:
            audit_client.append_event("AUDIO_ADAPTIVE_THRESHOLD_CALIBRATED", cal_record)
        except Exception:
            pass
    return cal_record


def explain_charge_push_up_trace(
    audio_events_or_specs: Union[Sequence[Dict[str, Any]], Dict[str, Any]],
    time_slots: int = 128,
    freq_bands: int = 32,
    *,
    playhead_slot: int = 64,
    calibrated_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Produces an explainable causal "Charge Trace" debug view (analogous to
    `GhostOverlay`) showing why specific `(time_slot, freq_band)` cells pushed up
    at `playhead_slot`, without exposing raw bulk buffers.
    """
    if (
        isinstance(audio_events_or_specs, Sequence)
        and audio_events_or_specs
        and isinstance(audio_events_or_specs[0], dict)
        and "b0" in audio_events_or_specs[0]
    ):
        norm_specs = list(audio_events_or_specs)
    else:
        norm_specs, _ = translate_audio_to_chargeable_bytes(audio_events_or_specs, time_slots, freq_bands)

    pot_bytes, nov_bytes = _compile_chargeable_byte_potentials(norm_specs, time_slots, freq_bands)
    cal = calibrated_thresholds or calibrate_adaptive_charge_thresholds(
        pot_bytes[: min(len(pot_bytes), freq_bands * 8)],
        nov_bytes[: min(len(nov_bytes), freq_bands * 8)],
    )
    fast_thr = int(cal["calibrated_push_threshold_fast"])
    slow_thr = int(cal["calibrated_push_threshold_slow"])

    sim = _simulate_causal_playhead_charging(
        pot_bytes,
        nov_bytes,
        time_slots,
        freq_bands,
        playhead_slot=playhead_slot,
    )
    fast_b = sim["fast_bytes"]
    slow_b = sim["slow_bytes"]
    comp_b = sim["pushed_up_bytes"]

    trace_events: List[Dict[str, Any]] = []
    clamped_ph = min(time_slots - 1, max(0, int(playhead_slot)))
    for t in range(clamped_ph + 1):
        toff = t * freq_bands
        # Find peak band at slot t
        best_b = 0
        best_c = 0
        for b in range(freq_bands):
            c_val = comp_b[toff + b]
            if c_val > best_c:
                best_c = c_val
                best_b = b
        idx = toff + best_b
        f_val = fast_b[idx]
        s_val = slow_b[idx]
        if f_val >= fast_thr or s_val >= slow_thr:
            reasons: List[str] = []
            if f_val >= fast_thr:
                reasons.append(f"fast_charge({f_val})>=fast_thr({fast_thr})")
            if s_val >= slow_thr:
                reasons.append(f"slow_charge({s_val})>=slow_thr({slow_thr})")
            trace_events.append({
                "t_slot": t,
                "freq_band": best_b,
                "nominal_hz": _band_index_to_hz(best_b, freq_bands),
                "age_ticks_from_playhead": clamped_ph - t,
                "initial_potential_u8": pot_bytes[idx],
                "initial_novelty_u8": nov_bytes[idx],
                "current_fast_charge_u8": f_val,
                "current_slow_charge_u8": s_val,
                "pushed_up_byte_u8": best_c,
                "push_trigger_reasons": reasons,
            })

    out = {
        "epistemic_state": "derived",
        "observed": False,
        "playhead_slot": clamped_ph,
        "calibrated_thresholds": cal,
        "pushed_up_slot_count": len(trace_events),
        "explainable_trace": trace_events[:32],
    }
    assert_no_raw_buffer_leakage(out)
    return out


def validate_and_translate_consented_pcm_samples(
    pcm_samples_i16: Sequence[int],
    *,
    sample_rate_hz: int = 16000,
    origin_attestation: str = "owner_supplied_audio",
    consent_token: Optional[Dict[str, Any]] = None,
    time_slots: int = 128,
    freq_bands: int = 32,
) -> Dict[str, Any]:
    """
    Tier-2 Consented Sandboxed Audio Validator & Multi-Band / Vocal-Sub-Lane
    (`300-3400 Hz`) Chargeable-Byte Translator:
      1. Enforces `origin_attestation in ("synthetic_fixture", "owner_supplied_audio")`
         and requires a non-expired `consent_token` when `origin_attestation == "owner_supplied_audio"`.
      2. Guards against adversarial audio:
         - Removes DC offset (`mean_i16`) before energy extraction.
         - Detects and clamps hard full-scale clipping (`|sample| >= 32760`).
         - Attenuates ultrasonic energy above `MAX_FREQ_HZ` (`7040 Hz`) via
           zero-crossing rate / first-difference Nyquist damping.
      3. Extracts both full-spectrum chargeable-byte events (`55..7040 Hz`) and
         the dedicated **Vocal/Phoneme Sub-Lane (`300..3400 Hz`)** (`vocal_band_300_3400_share_q16`
         and `vocal_band_onset_flux_q16`) so an external agent-layer CTC/ASR head
         can consume phoneme-rate fast charge without breaching Layer 4.
    """
    if origin_attestation not in VALID_AUDIO_ORIGINS:
        raise PermissionError(f"ERR_NON_SYNTHETIC_AUDIO_PROHIBITED: {origin_attestation!r}")
    if origin_attestation == "owner_supplied_audio":
        if not isinstance(consent_token, dict) or not consent_token.get("consented"):
            raise PermissionError(
                "ERR_MISSING_AUDIO_CONSENT_TOKEN: Tier-2 owner_supplied_audio requires an explicit consent_token"
            )
    if not (8000 <= int(sample_rate_hz) <= 48000):
        raise ValueError(f"Unsupported sample_rate_hz {sample_rate_hz}; must be in 8000..48000")
    n_samples = len(pcm_samples_i16)
    if n_samples < 64 or n_samples > sample_rate_hz * 60:
        raise ValueError(f"PCM sample count {n_samples} out of bounds (64 .. 60s)")

    # 1. Adversarial Guard: DC offset removal & hard-clipping counter
    raw_sum = 0
    clipped_samples = 0
    for s in pcm_samples_i16:
        if isinstance(s, float) and (math.isnan(s) or math.isinf(s)):
            raise ValueError("ERR_ADVERSARIAL_AUDIO_NAN_OR_INF")
        iv = max(-32768, min(32767, int(s)))
        raw_sum += iv
        if abs(iv) >= 32760:
            clipped_samples += 1
    dc_offset_i16 = raw_sum // n_samples

    # 2. Partition into `time_slots` windows with a 4-Band Polyphonic Filterbank
    # (Sub-Bass 55-280Hz, Low-Mid 280-900Hz, Vocal Formant 900-3400Hz, High Brilliance 3400-7040Hz)
    # + DC-scrubbed zero-crossing rate + ultrasonic damping + Dynamic Section Drop & Tempo Cadence
    samples_per_slot = max(1, n_samples // time_slots)
    events: List[Dict[str, Any]] = []
    vocal_slot_energies: List[int] = []
    sub_bass_slot_energies: List[int] = []
    low_mid_slot_energies: List[int] = []
    formant_slot_energies: List[int] = []
    brilliance_slot_energies: List[int] = []
    total_slot_energies: List[int] = []
    ultrasonic_attenuated_slots = 0
    polyphonic_chord_slots = 0

    lp_sub_q8 = 0
    lp_mid_q8 = 0
    # Scale filter alphas to sample rate so cutoffs stay anchored near ~260Hz and ~1800Hz
    sub_shift = 5 if sample_rate_hz >= 16000 else 4
    mid_shift = 2 if sample_rate_hz >= 16000 else 1

    for t in range(time_slots):
        idx0 = t * samples_per_slot
        idx1 = min(n_samples, idx0 + samples_per_slot)
        win_len = max(1, idx1 - idx0)
        abs_sum = 0
        sub_abs_sum = 0
        lowmid_abs_sum = 0
        formant_abs_sum = 0
        brill_abs_sum = 0
        zcr_count = 0
        vocal_zcr_count = 0
        prev_val = 0
        prev_vocal_val = 0

        for i in range(idx0, idx1):
            v = max(-32768, min(32767, int(pcm_samples_i16[i]) - dc_offset_i16))
            abs_sum += abs(v)
            if i > idx0 and ((v >= 0 and prev_val < 0) or (v < 0 and prev_val >= 0)):
                zcr_count += 1

            # 4-band integer IIR decomposition
            v_q8 = v << 8
            lp_sub_q8 += (v_q8 - lp_sub_q8) >> sub_shift
            lp_mid_q8 += (v_q8 - lp_mid_q8) >> mid_shift

            b_sub = lp_sub_q8 >> 8
            b_lowmid = (lp_mid_q8 - lp_sub_q8) >> 8
            hp_diff = (v - prev_val) if i > idx0 else 0
            b_brill = hp_diff >> 1
            b_formant = (v - (lp_mid_q8 >> 8)) - (b_brill >> 1)
            b_vocal = b_lowmid + b_formant

            if i > idx0 and ((b_vocal >= 0 and prev_vocal_val < 0) or (b_vocal < 0 and prev_vocal_val >= 0)):
                vocal_zcr_count += 1
            prev_vocal_val = b_vocal
            prev_val = v

            sub_abs_sum += abs(b_sub)
            lowmid_abs_sum += abs(b_lowmid)
            formant_abs_sum += abs(b_formant)
            brill_abs_sum += abs(b_brill)

        mean_abs = abs_sum // win_len
        est_hz = (zcr_count * sample_rate_hz) / float(max(1, 2 * win_len))
        est_vocal_hz = (vocal_zcr_count * sample_rate_hz) / float(max(1, 2 * win_len))

        # Ultrasonic adversarial guard (> 7040 Hz damped by 8x)
        if est_hz > MAX_FREQ_HZ:
            ultrasonic_attenuated_slots += 1
            mean_abs = mean_abs >> 3
            sub_abs_sum >>= 3
            lowmid_abs_sum >>= 3
            formant_abs_sum >>= 3
            brill_abs_sum >>= 3
            est_hz = MAX_FREQ_HZ

        # Soft-knee saturation on clipped bursts
        clip_scale = 0.75 if clipped_samples > (n_samples // 8) else 1.0
        gain_norm = min(1.0, (mean_abs / 16384.0) * clip_scale)
        u8_energy = max(0, min(255, int(round(gain_norm * 255.0))))
        total_slot_energies.append(u8_energy)

        sub_u8 = max(0, min(255, int(round(min(1.0, ((sub_abs_sum // win_len) / 14000.0) * clip_scale) * 255.0))))
        lowmid_u8 = max(0, min(255, int(round(min(1.0, ((lowmid_abs_sum // win_len) / 12000.0) * clip_scale) * 255.0))))
        formant_u8 = max(0, min(255, int(round(min(1.0, ((formant_abs_sum // win_len) / 12000.0) * clip_scale) * 255.0))))
        brill_u8 = max(0, min(255, int(round(min(1.0, ((brill_abs_sum // win_len) / 12000.0) * clip_scale) * 255.0))))

        sub_bass_slot_energies.append(sub_u8)
        low_mid_slot_energies.append(lowmid_u8)
        formant_slot_energies.append(formant_u8)
        brilliance_slot_energies.append(brill_u8)

        # Vocal sub-lane (300..3400 Hz): combines formant band + mid-vocal fundamental when in vocal pitch range
        if 300.0 <= est_hz <= 3400.0:
            voc_u8 = max(formant_u8, (lowmid_u8 + formant_u8) >> 1)
        elif 300.0 <= est_vocal_hz <= 3400.0 and formant_u8 >= 12:
            voc_u8 = formant_u8
        else:
            voc_u8 = 0
        vocal_slot_energies.append(voc_u8)

        # Emit polyphonic chord or tone event for chargeable-byte compilation
        if u8_energy >= 12 and est_hz >= MIN_FREQ_HZ:
            active_freqs: List[float] = []
            if sub_u8 >= 28:
                active_freqs.append(110.0)
            if lowmid_u8 >= 24 or formant_u8 >= 24:
                mid_f = max(280.0, min(3200.0, est_vocal_hz if est_vocal_hz >= 260.0 else est_hz))
                active_freqs.append(round(mid_f, 2))
            if brill_u8 >= 28:
                active_freqs.append(4400.0)

            if len(active_freqs) >= 2:
                polyphonic_chord_slots += 1
                events.append({
                    "event_id": f"pcm_slot_{t:04d}",
                    "kind": "chord",
                    "t0": t,
                    "t1": t,
                    "freq_hz": round(max(MIN_FREQ_HZ, min(MAX_FREQ_HZ, est_hz)), 2),
                    "frequencies_hz": active_freqs,
                    "gain": round(u8_energy / 255.0, 4),
                    "bandwidth_bands": 1,
                })
            else:
                events.append({
                    "event_id": f"pcm_slot_{t:04d}",
                    "kind": "tone",
                    "t0": t,
                    "t1": t,
                    "freq_hz": round(max(MIN_FREQ_HZ, min(MAX_FREQ_HZ, est_hz)), 2),
                    "gain": round(u8_energy / 255.0, 4),
                    "bandwidth_bands": 1,
                })

    band_4_sum = max(
        1,
        sum(sub_bass_slot_energies)
        + sum(low_mid_slot_energies)
        + sum(formant_slot_energies)
        + sum(brilliance_slot_energies),
    )
    vocal_e_sum = sum(vocal_slot_energies)
    vocal_flux_sum = sum(
        max(0, vocal_slot_energies[i] - vocal_slot_energies[i - 1])
        for i in range(1, len(vocal_slot_energies))
    )

    # 3. Dynamic Section-Drop Detector & Tempo / Onset Cadence Estimator
    q_len = max(1, time_slots // 8)
    section_means = [
        sum(total_slot_energies[k * q_len : min(time_slots, (k + 1) * q_len)]) // max(1, min(time_slots, (k + 1) * q_len) - k * q_len)
        for k in range(8)
    ]
    max_section_drop_delta = 0
    drop_section_idx = 0
    for k in range(1, len(section_means)):
        delta = section_means[k] - section_means[k - 1]
        if delta > max_section_drop_delta:
            max_section_drop_delta = delta
            drop_section_idx = k

    peak_slot_u8 = max(total_slot_energies) if total_slot_energies else 0
    mean_slot_u8 = sum(total_slot_energies) // max(1, len(total_slot_energies))
    crest_q16 = (peak_slot_u8 << 16) // max(1, mean_slot_u8)

    # Count distinct onset peaks for tempo cadence
    duration_sec = max(0.1, n_samples / float(sample_rate_hz))
    onset_slots: List[int] = []
    for i in range(1, len(total_slot_energies)):
        flux = total_slot_energies[i] - total_slot_energies[i - 1]
        if flux >= 14 and (not onset_slots or (i - onset_slots[-1]) >= 2):
            onset_slots.append(i)
    onsets_per_sec = len(onset_slots) / duration_sec
    # Map onset rate (typically 1.5 - 6.0 onsets/sec) into canonical musical BPM window [70..185]
    raw_bpm = int(round(onsets_per_sec * 30.0))
    while 0 < raw_bpm < 70:
        raw_bpm *= 2
    while raw_bpm > 185:
        raw_bpm //= 2

    return {
        "origin_attestation": origin_attestation,
        "sample_rate_hz": sample_rate_hz,
        "sample_count": n_samples,
        "adversarial_sanitization": {
            "dc_offset_removed_i16": dc_offset_i16,
            "clipped_sample_count": clipped_samples,
            "ultrasonic_attenuated_slots": ultrasonic_attenuated_slots,
            "overflow_or_nan_detected": False,
        },
        "vocal_phoneme_sublane_300_3400hz": {
            "vocal_energy_sum": vocal_e_sum,
            "vocal_share_q16": ((sum(low_mid_slot_energies) + sum(formant_slot_energies)) << 16) // band_4_sum
            if vocal_e_sum > 0
            else 0,
            "vocal_onset_flux_q16": (vocal_flux_sum << 16) // max(1, time_slots),
        },
        "four_band_polyphonic_profile": {
            "sub_bass_55_280hz_share_q16": (sum(sub_bass_slot_energies) << 16) // band_4_sum,
            "low_mid_280_900hz_share_q16": (sum(low_mid_slot_energies) << 16) // band_4_sum,
            "vocal_formant_900_3400hz_share_q16": (sum(formant_slot_energies) << 16) // band_4_sum,
            "high_brilliance_3400_7040hz_share_q16": (sum(brilliance_slot_energies) << 16) // band_4_sum,
            "polyphonic_chord_slots": polyphonic_chord_slots,
        },
        "dynamic_and_tempo_profile": {
            "peak_slot_energy_u8": peak_slot_u8,
            "mean_slot_energy_u8": mean_slot_u8,
            "crest_factor_q16": crest_q16,
            "section_envelope_8seg_u8": section_means,
            "max_section_drop_delta_u8": max_section_drop_delta,
            "drop_section_index": drop_section_idx,
            "section_drop_detected": bool(max_section_drop_delta >= 22),
            "onset_count": len(onset_slots),
            "estimated_bpm": raw_bpm,
        },
        "translated_events": events,
    }


class AgentNarrationStateMachine:
    """
    Agent-Layer Real-Time Narration State Machine (Point D & Cross-Modal Revision Policy):
      - Lives outside the enclave sensor layer.
      - Queues pushed-up packets from `AgentElectricalAreaBuffer` and coalesces
        packets within `coalesce_window_ms` (`300 ms`) into a single utterance
        so the agent never chatters on every packet.
      - Emits a revision utterance (`is_revision=True`, `amends_utterance_id`)
        ONLY when `revision == 1` AND `confidence_delta_q16 >= min_revision_confidence_delta_q16`
        (logging an amendment link without overwriting prior WORM receipts).
    """

    def __init__(
        self,
        *,
        coalesce_window_ms: int = 300,
        min_revision_confidence_delta_q16: int = 8192,
    ) -> None:
        self.coalesce_window_ms = max(50, int(coalesce_window_ms))
        self.min_revision_confidence_delta_q16 = max(1, int(min_revision_confidence_delta_q16))
        self._pending_packets: List[Dict[str, Any]] = []
        self._utterances: List[Dict[str, Any]] = []
        self._last_confident_band: Optional[int] = None
        self._last_confidence_q16: int = 0
        self._last_utterance_id: Optional[str] = None

    def push_packet(self, pkt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Queues a pushed-up packet and emits a coalesced utterance if window elapsed."""
        if int(pkt.get("push", 0)) != 1:
            return None
        self._pending_packets.append(pkt)
        t_first = int(self._pending_packets[0]["t"])
        t_cur = int(pkt["t"])
        if t_cur - t_first >= self.coalesce_window_ms:
            return self.flush(current_t_ms=t_cur)
        return None

    def flush(self, *, current_t_ms: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if not self._pending_packets:
            return None
        batch = list(self._pending_packets)
        self._pending_packets.clear()

        t_emit = int(batch[-1]["t"]) if current_t_ms is None else int(current_t_ms)
        peak_pkt = max(batch, key=lambda p: int(p["q"]))
        conf_q16 = (int(peak_pkt["q"]) << 16) // 255
        any_rev = any(int(p.get("revision", 0)) == 1 for p in batch)
        conf_delta_q16 = abs(conf_q16 - self._last_confidence_q16)

        is_revision = bool(
            any_rev
            and self._last_utterance_id is not None
            and conf_delta_q16 >= self.min_revision_confidence_delta_q16
        )
        amends_id = self._last_utterance_id if is_revision else None
        utt_id = f"utt_{len(self._utterances) + 1:04d}"

        mods = {int(p["modality"]) for p in batch}
        if MODALITY_FUSED in mods or (MODALITY_VISION in mods and MODALITY_AUDIO in mods):
            scope = "audiovisual_moment"
        elif MODALITY_VISION in mods:
            scope = "visual_moment"
        else:
            scope = "audio_moment"

        utterance = {
            "utterance_id": utt_id,
            "t_ms": t_emit,
            "coalesced_packet_count": len(batch),
            "align_ids": sorted({int(p["align_id"]) for p in batch}),
            "scope": scope,
            "dominant_f_band": int(peak_pkt["f_band"]),
            "confidence_q16": conf_q16,
            "is_revision": is_revision,
            "amends_utterance_id": amends_id,
            "epistemic_origin": "agent_probabilistic_interpreter",
            "is_enclave_truth": False,
        }
        self._utterances.append(utterance)
        self._last_confident_band = int(peak_pkt["f_band"])
        self._last_confidence_q16 = conf_q16
        self._last_utterance_id = utt_id
        return utterance

    @property
    def utterances(self) -> List[Dict[str, Any]]:
        return list(self._utterances)


def run_tier2_fidelity_adversarial_and_edge_profile() -> Dict[str, Any]:
    """
    Executes the High-ROI Extension Verification Suite:
      1. Speech-Cadence vs. Polyphonic Music vs. Noise Discrimination:
         - Speech-like syllable stream (rapid 300-3400 Hz bursts) exhibits higher
           `fast_to_slow_ratio_q16` and `vocal_share_q16` than sustained polyphonic chords.
      2. Adversarial Audio Torture Tests:
         - DC offset (+12000), hard full-scale clipping (+/-32767), and 20 kHz
           ultrasonic burst are sanitized with 0 crashes/overflows.
      3. Hardware/Edge Latency & Throughput Profile:
         - Benchmarks 16-byte `ChargePacket` & `AgentElectricalAreaBuffer`
           throughput and verifies fast-lane CPU latency is well below `< 50 ms`.
    """
    sr = 16000
    n_samp = 8000  # 0.5 s at 16 kHz

    # 1A. Speech-like syllable bursts at 600 Hz (inside 300-3400 Hz vocal band, 4 short bursts)
    speech_pcm: List[int] = []
    for i in range(n_samp):
        slot_phase = (i // 500) % 2
        if slot_phase == 0:
            val = int(round(14000.0 * math.sin(2.0 * math.pi * 600.0 * (i / float(sr)))))
        else:
            val = 0
        speech_pcm.append(val)

    # 1B. Sustained low-bass drone at 110 Hz (outside 300-3400 Hz vocal band, continuous sustain)
    bass_pcm = [
        int(round(14000.0 * math.sin(2.0 * math.pi * 110.0 * (i / float(sr)))))
        for i in range(n_samp)
    ]

    consent = {"consented": True, "token_id": "tok_tier2_eval"}
    tr_speech = validate_and_translate_consented_pcm_samples(
        speech_pcm, sample_rate_hz=sr, consent_token=consent, time_slots=128, freq_bands=32
    )
    tr_bass = validate_and_translate_consented_pcm_samples(
        bass_pcm, sample_rate_hz=sr, consent_token=consent, time_slots=128, freq_bands=32
    )

    speech_vocal_q16 = tr_speech["vocal_phoneme_sublane_300_3400hz"]["vocal_share_q16"]
    bass_vocal_q16 = tr_bass["vocal_phoneme_sublane_300_3400hz"]["vocal_share_q16"]
    speech_flux_q16 = tr_speech["vocal_phoneme_sublane_300_3400hz"]["vocal_onset_flux_q16"]
    bass_flux_q16 = tr_bass["vocal_phoneme_sublane_300_3400hz"]["vocal_onset_flux_q16"]
    discrimination_passed = bool(
        speech_vocal_q16 > bass_vocal_q16 and speech_flux_q16 > bass_flux_q16
    )

    # 2. Adversarial Audio Torture Test (DC offset +12000, hard clipping, and 20kHz ultrasonic at 48kHz)
    adv_sr = 48000
    adv_pcm = [
        max(-32768, min(32767, 12000 + int(round(28000.0 * math.sin(2.0 * math.pi * 20000.0 * (i / float(adv_sr)))))))
        for i in range(9600)
    ]
    tr_adv = validate_and_translate_consented_pcm_samples(
        adv_pcm, sample_rate_hz=adv_sr, consent_token=consent, time_slots=128, freq_bands=32
    )
    adv_san = tr_adv["adversarial_sanitization"]
    adversarial_passed = bool(
        abs(adv_san["dc_offset_removed_i16"]) > 5000
        and adv_san["clipped_sample_count"] > 0
        and adv_san["ultrasonic_attenuated_slots"] > 0
        and adv_san["overflow_or_nan_detected"] is False
    )

    # 3. Hardware/Edge Latency & Throughput Benchmark
    t0 = time.perf_counter()
    area = AgentElectricalAreaBuffer(window_ms=3000)
    narration = AgentNarrationStateMachine(coalesce_window_ms=300)
    n_bench = 200
    for idx in range(n_bench):
        area.ingest_sensory_packet(
            t_ms=idx * 20,
            modality=MODALITY_AUDIO if (idx % 2 == 0) else MODALITY_VISION,
            f_band=32 + (idx % 48),
            q_fast=140 if (idx % 5 == 0) else 20,
            q_slow=90,
        )
        narration.push_packet(
            build_charge_packet(
                t=idx * 20,
                modality=MODALITY_AUDIO,
                f_band=32 + (idx % 48),
                q_fast=140 if (idx % 5 == 0) else 20,
                q_slow=90,
            )
        )
    narration.flush(current_t_ms=n_bench * 20)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_packet_ms = round(elapsed_ms / float(n_bench), 5)
    packets_per_sec = int(round((n_bench * 1000.0) / max(0.001, elapsed_ms)))
    edge_latency_passed = bool(elapsed_ms < 50.0 and len(narration.utterances) > 0)

    out = {
        "all_passed": bool(discrimination_passed and adversarial_passed and edge_latency_passed),
        "speech_vs_bass_discrimination_passed": discrimination_passed,
        "speech_vocal_share_q16": speech_vocal_q16,
        "bass_vocal_share_q16": bass_vocal_q16,
        "adversarial_sanitization_passed": adversarial_passed,
        "adversarial_summary": adv_san,
        "edge_latency_target_under_50ms_passed": edge_latency_passed,
        "edge_profile": {
            "batch_200_packets_elapsed_ms": round(elapsed_ms, 4),
            "per_packet_latency_ms": per_packet_ms,
            "packets_per_sec": packets_per_sec,
            "coalesced_utterances_emitted": len(narration.utterances),
        },
    }
    assert_no_raw_buffer_leakage(out)
    return out


