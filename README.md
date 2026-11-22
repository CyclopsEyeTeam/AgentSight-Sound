# Electric Sight & Sound for Agents

**Deterministic Pixel & Chargeable-Byte Audio Perception Enclave for Autonomous AI Agents**

![Electric Sight & Sound — Verified Enclave Emblem](emblem/drawing.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](pyproject.toml)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Stdlib%20%2B%20C11%20AVX2)-success.svg)](Makefile)

**Electric Sight & Sound** is a baseline, zero-dependency, open-source perception engine that gives AI agents **deterministic eyes and ears** — built as a physical sensor enclave rather than a black-box neural vibe.

It is designed to be **connected to an infinite number of upstream producers and downstream agent workflows** (LLMs/VLMs, game engines, DAWs, robotics simulators, UI renderers, accessibility tools, neuromorphic event buses, and media pipelines) via a compact **16-byte wire protocol (`ChargePacket`)** and cryptographically countersigned **Observation Receipts**.

---

## Why Electric Sight & Sound?

1. **Chargeable-Byte Audio ("Bytes Get Charged Up During Play")**:
   - Audio is translated into a 2D time-frequency matrix of **chargeable bytes (`0..255`)**.
   - Before playback (`playhead_slot == -1`), all chargeable bytes rest at `0`.
   - As the playhead sweeps across the sound, bytes at the active time slot are **pushed up (charged up)** across two multi-scale temporal lanes:
     - **Fast Charge Lane (`q_fast`, ~50 ms half-life)**: Captures sharp transients, consonant/phoneme onsets, drum hits, and rapid pitch shifts.
     - **Slow Charge Lane (`q_slow`, ~1.5 s half-life)**: Acts as a built-in short-term capacitor retaining harmonic contour, sustained chords, and mood without requiring extra RNN state.
   - **Refractory & Saturation Guards**: Per-band `0..255` ceiling and refractory cooldown ticks prevent loud bursts or adversarial clipping from flooding agent attention.
2. **Deterministic Electric Vision (`128x128` to `1920x1080`) & Companion Continuous Desktop Sight (`desktop_sight.py`)**:
   - 10 closed-form integer/`Q16.16` spatial operators + native C11/AVX2 (`prod_gate_native_simd.c`) & Rust (`#![forbid(unsafe_code)]`, `prod_gate_native_simd.rs`) SIMD operators (`10,000/10,000` differential fuzz parity, `0` divergences).
   - Continuous streaming ring-buffer (`ContinuousElectricEyeStream`) with `explicit_bzero` zeroization on frame eviction, $16\times16$ dirty-tile bitmasks, automatic `verified -> stale` epistemic decay watchdogs, and honest `ingest_frame(..., origin_attestation=...)` pass-through.
   - **Provenance Posture & Companion Exception (2026-11-04)**: Generic live capture (`live_webcam_capture`, `live_screen_capture`, `live_microphone_capture`) stays deterministically refused (`ERR_NON_SYNTHETIC_INPUT_PROHIBITED` / `ERR_NON_SYNTHETIC_AUDIO_PROHIBITED`), **except the Companion's own glance and system-audio listen, on the owner's word**:
     - `origin_attestation="companion_desktop_glance"` ([`desktop_sight.py`](desktop_sight.py), `DesktopSight().glance()` / `.close()`): captures one root window frame into memory via pipe (`import -window root png:-`, never written to disk), zeroes the PNG buffer immediately after decode, measures a $3\times3$ box average (light, colour name, brightest region) plus $512\times512$ `ContinuousElectricEyeStream` temporal transitions, and returns `eyeLine`, plain words, scalar `measures`, and a Layer A `receipt` (`pixel_sha256`, `dhash64`, `diff_replay_digest`, `tick`) with zero pixels, zero OCR, and zero semantics. Work-chat MCP tools (`electric_vision_mcp.py:112`) intentionally exclude `companion_desktop_glance` from their `origin_attestation` enum so Work-chat agents cannot claim it.
     - `origin_attestation="companion_system_audio"` ([`desktop_ear.py`](desktop_ear.py), [`validate_and_translate_consented_pcm_samples()`](electric_audio_agent_bridge.py)): accepts the Companion's in-memory 16 kHz mono system-audio monitor stream (`parec` from default sink `.monitor`, never the microphone, zeroed on slot eviction and teardown) via `ContinuousSystemAudioMonitor` (12 s sliding ring buffer, `< 5 ms` warm reads) and `ContinuousElectricEarStream` (8-window rolling `stabilized_bpm` and `ear_regime`: `initializing | silence | steady | building | drop | breakdown`) with an explicit consent token (`scope="Companion listening to system audio, measurements only, never saved"`).
3. **Three-Clock Multimodal Fusion, Horizontal Charge Spread & 8-Byte `VibrationPacket` (Option B)**:
   - Packs visual (`modality=0`), audio (`modality=1`), fused audiovisual (`modality=2`), and fused audiovisual+tactile (`modality=3`) events into a **16-byte binary `ChargePacket`**, alongside a parallel **8-byte `VibrationPacket` (`>IHBB`: `t`, `align_id`, `q_vibe`, `h_spread`)**.
   - Models lateral spectral leakage via [`apply_horizontal_spread()`](electric_audio_agent_bridge.py) (wide `0.40–0.50` spread for `20–120 Hz` bass down to focused `0.05–0.10` for treble) and adds **Operators 11–13** (`op_horizontal_charge_spread_index_q16`, `op_bass_vibration_energy_u8`, `op_tactile_salience_q16`).
   - Time-aligns visual, auditory, and tactile push-ups within `100 ms` `align_id` windows inside [`AgentElectricalAreaBuffer`](electric_audio_agent_bridge.py) (with a 30% vibration budget cap) and drives external motors safely through [`HapticActuator`](electric_audio_agent_bridge.py) (6 safety gates: consent, rate limiter, `<=80%` intensity cap, `30s` duration auto-stop, `<100ms` kill switch, and `20 Hz` infrasound floor).
4. **Strict Epistemic Hygiene (`SEMANTIC_BOUNDARY_CONTRACT`)**:
   - **Enclave Sensor Layer (`is_enclave_truth=True`, `claims_semantic_meaning=False`)**: Emits deterministic pixel/charge measurements, explainable push-up traces, and HMAC-SHA256 countersigned receipts (`ObservationReceipt`).
   - **Agent Interpreter Layer (`is_enclave_truth=False`)**: External LLM/ASR/mood interpretations and coalesced narrations ([`AgentNarrationStateMachine`](electric_audio_agent_bridge.py)) are wrapped in [`create_agent_interpretation_envelope()`](electric_audio_agent_bridge.py) so stochastic hypotheses are never confused with sensor truth.

---

## Architecture Diagram

```mermaid
flowchart LR
    subgraph Producers["Infinite Upstream Producers"]
        V["Vector / UI / Game / Pixel Frames\n(128x128 .. 1920x1080)"]
        A1["Synth / MIDI / Tone Alphabet V0\n(tone, chord, sweep, pulse)"]
        A2["Consented PCM / WAV / MP3 Streams\n(Tier-2 4-Band Polyphonic Filterbank)"]
    end

    subgraph Enclave["Deterministic Perception Enclave (is_enclave_truth = True)"]
        EV["Electric Vision Session\n10 Closed-Form + AVX2 SIMD Ops"]
        EA["Electric Audio Session\nChargeable Bytes + Fast/Slow Push-Up"]
        HSM["Out-of-Process Daemons\nHSM Receipt Signer + WORM Audit Ledger"]
    end

    subgraph Fusion["Real-Time Multimodal Fusion & Agent Layer"]
        PKT["16-Byte ChargePacket\n(100ms align_id Window)"]
        AREA["AgentElectricalAreaBuffer\n(3s Working Memory + Decay)"]
        NARR["AgentNarrationStateMachine\n(300ms Coalescing + Revisions)"]
    end

    V --> EV
    A1 --> EA
    A2 --> EA
    EV --> HSM
    EA --> HSM
    EV --> PKT
    EA --> PKT
    PKT --> AREA --> NARR
```

---

## Quick Start (Zero External Python Dependencies)

Requires **Python 3.10+** on Linux (and `gcc` if rebuilding `libprod_gate_simd.so`).

```bash
# 1. Build the native C11/AVX2 SIMD library, run all 31 unit test suites, and run CLI self-tests
make all

# 2. Run the universal connector demonstration
make demo

# 3. Run the Electric Audio & Sight+Sound CLI self-test
python3 electric_audio_cli.py --no-color --plain --mode focus selftest

# 4. Run the Electric Vision CLI self-test
python3 electric_vision_cli.py --no-color --plain --mode focus selftest

# 5. Inspect an explainable causal push-up trace on Tone Alphabet audio
python3 electric_audio_cli.py --no-color --plain charge-trace --tone-text "HELLO WORLD" --top-k 5
```

---

## Connecting Anything to Electric Sight & Sound

Because the enclave exposes clean Python session managers, a 16-byte binary packet codec, and JSON/ASCII CLI tools, you can wire it into any system:

### 1. Connect Any PCM / WAV / Decoded Audio Stream (Tier-2 4-Band Polyphonic Filterbank)

```python
from electric_audio_agent_bridge import (
    ElectricAudioAgentSession,
    validate_and_translate_consented_pcm_samples,
    explain_charge_push_up_trace,
)

# Feed 16-bit mono PCM samples (8kHz..48kHz) with an explicit consent token
tier2 = validate_and_translate_consented_pcm_samples(
    pcm_samples_i16,
    sample_rate_hz=16000,
    origin_attestation="owner_supplied_audio",
    consent_token={"consented": True, "token_id": "tok_session_01"},
    time_slots=256,
    freq_bands=64,
)

# Inspect 4-band polyphonic split (Sub-Bass 55-280Hz, Low-Mid 280-900Hz,
# Vocal Formant 900-3400Hz, High Brilliance 3.4-7kHz) and section-drop detection
print(tier2["four_band_polyphonic_profile"])
print(tier2["dynamic_and_tempo_profile"])

with ElectricAudioAgentSession(default_grid=(256, 64)) as session:
    session.register_chargeable_audio(
        "src_custom_track", "v1", tier2["translated_events"],
        playhead_ratio=1.0, origin_attestation="owner_supplied_audio",
    )
    report = session.inspect_audio("src_custom_track", "v1", issue_observation_receipt=True)
```

### 2. Connect Multimodal Vision + Audio in Real Time (`16-Byte ChargePacket`)

```python
from electric_audio_agent_bridge import (
    MODALITY_VISION,
    MODALITY_AUDIO,
    AgentElectricalAreaBuffer,
    AgentNarrationStateMachine,
    pack_charge_packet_16bytes,
)

area = AgentElectricalAreaBuffer(window_ms=3000)
narrator = AgentNarrationStateMachine(coalesce_window_ms=300)

# Vision event at t=110ms + Audio transient at t=145ms -> fused into align_id=1!
area.ingest_sensory_packet(t_ms=110, modality=MODALITY_VISION, f_band=32, q_fast=145, q_slow=70)
res = area.ingest_sensory_packet(t_ms=145, modality=MODALITY_AUDIO, f_band=48, q_fast=180, q_slow=95)

wire_bytes = pack_charge_packet_16bytes(res["fused_packet"])  # 16 bytes ready for IPC/UDP/WebSocket
narrator.push_packet(res["fused_packet"])
utterance = narrator.flush(current_t_ms=450)
```

---

## Empirical Multi-Genre Benchmark (10 Real-World Tracks)

Tested across 10 real-world tracks spanning a cappella, hip-hop, nu-metal, acoustic folk, liquid drum & bass, synth-pop, gabber, 1960s soul, crunk, and pop on a `256×64` chargeable-byte grid:

| Genre / Archetype | Sub-Bass (`55–280Hz`) | Low-Mid (`280–900Hz`) | Vocal Formant (`900–3400Hz`) | Brilliance (`3.4–7kHz`) | Polyphonic Chords | Section Drop (`Δu8`) | Charge Regime | Centroid |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A Cappella & Whistling** | 20.5% | 48.7% | 23.0% | 7.8% | **0** *(monophonic)* | `False` (`8`) | `fast_transient_dominant` | `647 Hz` |
| **West Coast Hip-Hop (808s)** | **32.4%** | 39.8% | 19.4% | 8.3% | **156** | `False` (`14`) | `balanced_attack_sustain` | **`257 Hz`** |
| **Nu-Metal (Distorted Guitars)** | **33.4%** | 36.0% | 18.7% | 11.9% | **251** *(wall of sound)* | `False` (`17`) | `slow_sustained_memory` | `554 Hz` |
| **Intimate Acoustic Folk** | 35.5% | 44.9% | 14.0% | 5.6% | **1** *(sparse guitar)* | `False` (`7`) | `fast_transient_dominant` | `323 Hz` |
| **Liquid Drum & Bass (174 BPM)** | 29.5% | 38.4% | 20.1% | 12.0% | **184** | **`True` (`+100`!)** | `slow_sustained_memory` | `698 Hz` |
| **80s Synth-Pop** | 21.5% | 36.6% | 26.1% | **15.8%** | **28** | `False` (`9`) | `fast_transient_dominant` | `513 Hz` |
| **90s Gabber / Rave (~190 BPM)** | 26.1% | 42.6% | 20.2% | 11.1% | **231** | **`True` (`+51`)** | `slow_sustained_memory` | `440 Hz` |
| **1960s Soul & Horn Section** | **14.1%** *(vintage)* | 43.6% | **28.8%** *(highest)* | 13.5% | **85** | **`True` (`+27`)** | `balanced_attack_sustain` | **`1630 Hz`** |
| **2000s Crunk (Synth Lead + 808)** | 29.8% | 29.4% | 24.0% | **16.8%** *(highest)* | **235** | `False` (`9`) | `slow_sustained_memory` | `754 Hz` |
| **Modern Pop** | 32.7% | 37.8% | 18.6% | 10.9% | **238** | **`True` (`+27`)** | `slow_sustained_memory` | `475 Hz` |

---

## Repository Structure

| File | Description |
| :--- | :--- |
| [`electric_eye_harness.py`](electric_eye_harness.py) | Core 5-layer epistemic harness, `E1`–`E10` scenarios, and `C1`–`C9` pass criteria. |
| [`exec_gate_daemons.py`](exec_gate_daemons.py) | Out-of-process `AF_UNIX` Vault, Audit WORM Sink, and Quarantine/Kill-Switch daemons (`EXEC-GATE-1..3`). |
| [`prod_gate_enclave.py`](prod_gate_enclave.py) | Out-of-process HSM/KMS enclave, `NativeSimdOperatorEngine` (including fast AVX2 signal pass `run_fast_signal_operators_native`), PII canary scrubber, and WCAG validator (`PROD-GATE-1..4`). |
| [`prod_gate_native_simd.c`](prod_gate_native_simd.c) / [`prod_gate_native_simd.rs`](prod_gate_native_simd.rs) | Native C11/AVX2 + Rust (`#![forbid(unsafe_code)]`) SIMD operator library. |
| [`electric_vision_agent_bridge.py`](electric_vision_agent_bridge.py) | Multi-resolution (`128x128`..`1920x1080`) agent vision bridge & [`ContinuousElectricEyeStream`](electric_vision_agent_bridge.py) (`ingest_signal_frame`, `ingest_buffered_repeat`, `lazy_enclave=True` with on-demand proxy materialization, and `explicit_bzero` ring eviction). |
| [`signal_buffer.py`](signal_buffer.py) | [`ElectricSignalTempBuffer`](signal_buffer.py) — active per-source electric/visual signal cache (`raw_sha256` gate + memoized chromatic swatch extractor) eliminating redundant live pixel-to-mark translation on static or streaming windows. |
| [`x11_window_capture.py`](x11_window_capture.py) | [`X11WindowCapture`](x11_window_capture.py) — persistent Xlib `_NET_CLIENT_LIST` window lister & `libX11`/`libXcomposite` memory-only `explicit_bzero` window capture. |
| [`desktop_sight.py`](desktop_sight.py) / [`continuous_vision.py`](continuous_vision.py) / [`visual_sources.py`](visual_sources.py) | Continuous desktop sight (`DesktopSight`), source-pinned temporal field recording, and confined image/SVG/video/HTML loaders. |
| [`desktop_ear.py`](desktop_ear.py) | Continuous system-audio monitor (`DesktopEar`, `ContinuousSystemAudioMonitor`, `origin_attestation="companion_system_audio"`). |
| [`watch_together.py`](watch_together.py) | [`WatchTogetherSession`](watch_together.py) — combined Sight + Sound + Vibration watch-together engine with `align_id` 100 ms multimodal fusion and multi-resident attention lanes. |
| [`electric_vision_mcp.py`](electric_vision_mcp.py) / [`electric_sight_ear_mcp.py`](electric_sight_ear_mcp.py) | Zero-leakage stdio Model Context Protocol (MCP) servers for confined visual files and unified Sight + Ear + Watch-Together tools. |
| [`electric_vision_cli.py`](electric_vision_cli.py) / [`electric_audio_cli.py`](electric_audio_cli.py) | Standalone Electric Vision (`cyclops-ee-ctl`) and Electric Audio & Sight+Sound (`cyclops-ea-ctl`) CLIs. |
| [`electric_audio_agent_bridge.py`](electric_audio_agent_bridge.py) | Chargeable-Byte Electric Audio bridge, [`ContinuousElectricEarStream`](electric_audio_agent_bridge.py), [`ElectricSightAndSoundSession`](electric_audio_agent_bridge.py), 16-byte `ChargePacket`, 8-byte `VibrationPacket`, [`AgentElectricalAreaBuffer`](electric_audio_agent_bridge.py), and Tier-2 4-band PCM validator. |
| [`examples/connect_anything_demo.py`](examples/connect_anything_demo.py) | Universal connector script demonstrating Sight+Sound, Tier-2 PCM, and 16-byte `ChargePacket` fusion. |
| [`test_electric_eye_harness.py`](test_electric_eye_harness.py) / [`test_electric_audio_harness.py`](test_electric_audio_harness.py) / [`test_optional_backends.py`](test_optional_backends.py) / [`test_static_review.py`](test_static_review.py) / [`test_signal_buffer.py`](test_signal_buffer.py) | Complete 80+-test unit, differential SIMD fuzz, security gate, audio, and signal-buffer verification suite. |

---

## License

Released under the [MIT License](LICENSE).
