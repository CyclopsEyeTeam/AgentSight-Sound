#!/usr/bin/env python3
"""
examples/connect_anything_demo.py

Universal Connector Demonstration for Electric Sight & Sound.
Shows how to plug any external producer or consumer into the baseline
perception enclave in a few lines of Python:
  1. Visual Canvases / UI / Game Engines / Vector Diagrams -> Electric Sight
  2. Tone Alphabet / MIDI / Synth Events -> Chargeable-Byte Electric Audio
  3. Consented PCM / WAV / Music / Speech Streams -> Tier-2 4-Band Polyphonic Filterbank
  4. Real-Time Audiovisual Fusion -> 16-Byte ChargePacket + Electrical Area + Narration
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from electric_audio_agent_bridge import (
    MODALITY_AUDIO,
    MODALITY_VISION,
    SEMANTIC_BOUNDARY_CONTRACT,
    AgentElectricalAreaBuffer,
    AgentNarrationStateMachine,
    ElectricSightAndSoundSession,
    HapticActuator,
    create_agent_interpretation_envelope,
    explain_charge_push_up_trace,
    pack_charge_packet_16bytes,
    pack_vibration_packet_8bytes,
    validate_and_translate_consented_pcm_samples,
)


def main() -> None:
    print("=== 1. Multimodal Sight + Sound Session (Agents See and Hear Together) ===")
    with ElectricSightAndSoundSession() as av_session:
        av_report = av_session.inspect_sight_and_sound(
            "src_demo_sight",
            "src_demo_sound",
            "v1",
            audio_events_or_spec={"tone_text": "SIGHT AND SOUND"},
            playhead_ratio=0.9,
            issue_observation_receipts=True,
        )
        fusion = av_report["multimodal_fusion"]
        sight_rcpt = av_report["sight"]["render"]["observation_receipt_id"]
        sound_rcpt = av_report["sound"]["render"]["observation_receipt_id"]
        sound_digest = av_report["sound"]["audio_replay_digest"]
        print(f"  Modal Alignment    : {fusion['modal_alignment']}")
        print(f"  Sight Receipt ID   : {sight_rcpt} (Observed={av_report['sight']['render']['observed']})")
        print(f"  Sound Receipt ID   : {sound_rcpt} (Observed={av_report['sound']['render']['observed']})")
        print(f"  Joint Salience Q16 : {fusion['joint_salience_q16']}")

    print("\n=== 2. Tier-2 Consented PCM Stream (4-Band Polyphonic Filterbank + Drop Detector) ===")
    sr = 16000
    # Synthesize 0.5s of quiet intro followed by a polyphonic 3-band drop (110Hz sub + 660Hz vocal + 4400Hz brilliance)
    pcm_samples = []
    for i in range(sr // 2):
        t = i / float(sr)
        if i < (sr // 8):
            val = int(400.0 * math.sin(2.0 * math.pi * 440.0 * t))
        else:
            sub = 7000.0 * math.sin(2.0 * math.pi * 110.0 * t)
            voc = 6000.0 * math.sin(2.0 * math.pi * 660.0 * t)
            brill = 3500.0 * math.sin(2.0 * math.pi * 4400.0 * t)
            val = int(sub + voc + brill)
        pcm_samples.append(val)

    tier2 = validate_and_translate_consented_pcm_samples(
        pcm_samples,
        sample_rate_hz=sr,
        origin_attestation="owner_supplied_audio",
        consent_token={"consented": True, "token_id": "tok_demo_stream"},
        time_slots=128,
        freq_bands=32,
    )
    fb = tier2["four_band_polyphonic_profile"]
    dt = tier2["dynamic_and_tempo_profile"]
    print(f"  Polyphonic Chords  : {fb['polyphonic_chord_slots']} slots")
    print(f"  Section Drop Det.  : {dt['section_drop_detected']} (delta_u8=+{dt['max_section_drop_delta_u8']})")
    trace = explain_charge_push_up_trace(tier2["translated_events"], 128, 32, playhead_slot=100)
    print(f"  Explainable Trace  : {trace['pushed_up_slot_count']} pushed-up slots above adaptive threshold")

    print("\n=== 3. Three-Clock Electrical Area + Narration Coalescing + Semantic Boundary ===")
    area = AgentElectricalAreaBuffer(window_ms=3000)
    narrator = AgentNarrationStateMachine(coalesce_window_ms=300)

    # Vision and audio events arrive within the same 100ms align_id window (t=210ms and t=250ms)
    res_v = area.ingest_sensory_packet(t_ms=210, modality=MODALITY_VISION, f_band=40, q_fast=150, q_slow=80)
    res_a = area.ingest_sensory_packet(t_ms=250, modality=MODALITY_AUDIO, f_band=48, q_fast=175, q_slow=95)
    fused_pkt = res_a["fused_packet"]
    raw_16b = pack_charge_packet_16bytes(fused_pkt)
    narrator.push_packet(fused_pkt)
    utterance = narrator.flush(current_t_ms=550)

    interp_envelope = create_agent_interpretation_envelope(
        enclave_receipt_id=sound_rcpt,
        enclave_replay_digest=sound_digest,
        model_id="external-agent-narrator-v1",
        prompt_version="v1.0",
        interpreted_tokens=[utterance["scope"], "coincident_audiovisual_event"],
        confidence_q16=utterance["confidence_q16"],
    )
    print(f"  16-Byte Wire Packet: {raw_16b.hex()} ({len(raw_16b)} bytes, modality={fused_pkt['modality']} FUSED)")
    print(f"  Coalesced Utterance: {json.dumps(utterance, sort_keys=True)}")
    print(f"  Boundary Contract  : {SEMANTIC_BOUNDARY_CONTRACT['contract_id']} (is_enclave_truth={interp_envelope['is_enclave_truth']})")

    print("\n=== 4. Horizontal Charge Spread + 8-Byte VibrationPacket (Option B) + HapticActuator ===")
    res_vibe = area.ingest_vibration_packet(t_ms=265, q_vibe=220, h_spread=168)
    vpkt = res_vibe["ingested_vibration_packet"]
    fused_tactile = res_vibe["fused_tactile_packet"]
    raw_8b = pack_vibration_packet_8bytes(vpkt)
    actuator = HapticActuator(user_consented=True)
    haptic_out = actuator.drive(t_ms=265, q_vibe=vpkt["q_vibe"], h_spread=vpkt["h_spread"], freq_hz=60.0)
    print(f"  8-Byte Vibe Packet : {raw_8b.hex()} ({len(raw_8b)} bytes, align_id={vpkt['align_id']}, q_vibe={vpkt['q_vibe']}, h_spread={vpkt['h_spread']})")
    print(f"  Modality=3 Fusion  : modality={fused_tactile['modality']} ({fused_tactile['agent_description']} / {fused_tactile['physicality']['regime']})")
    print(f"  Haptic Actuator    : intensity={haptic_out['motor_intensity']} (cap={haptic_out['intensity_cap']}), pattern={haptic_out['motor_pattern']}")


if __name__ == "__main__":
    main()
