#!/usr/bin/env python3
"""
test_electric_audio_harness.py

Unit and Integration Verification Suite for the Chargeable-Byte Electric Audio
("Electric Ear") and Multimodal Sight+Sound stack.
"""

from __future__ import annotations

import json
import unittest

from electric_audio_agent_bridge import (
    MODALITY_AUDIO,
    MODALITY_FUSED,
    MODALITY_VISION,
    SEMANTIC_BOUNDARY_CONTRACT,
    AgentElectricalAreaBuffer,
    AgentNarrationStateMachine,
    ElectricAudioAgentSession,
    ElectricSightAndSoundSession,
    ablate_charge_lanes,
    build_charge_packet,
    calibrate_adaptive_charge_thresholds,
    create_agent_interpretation_envelope,
    explain_charge_push_up_trace,
    pack_charge_packet_16bytes,
    run_realtime_fusion_test_plan,
    run_tier2_fidelity_adversarial_and_edge_profile,
    tone_alphabet_text_to_events,
    unpack_charge_packet_16bytes,
    validate_and_translate_consented_pcm_samples,
)
from electric_audio_cli import run_cli
from electric_eye_harness import REFUSAL_CODE_SEMANTIC, assert_no_raw_buffer_leakage


class TestElectricAudioAndSightSoundHarness(unittest.TestCase):
    def test_tone_alphabet_v0_and_chargeable_byte_translation(self) -> None:
        plan = tone_alphabet_text_to_events("HELLO")
        self.assertEqual(plan["profile_id"], "az-log-220-880-v1")
        self.assertEqual(plan["event_count"], 5)
        # H, E, L, L, O -> indices 3 and 4 (L, L) have identical frequency but distinct time slots
        ev_l1 = plan["events"][2]
        ev_l2 = plan["events"][3]
        self.assertAlmostEqual(ev_l1["freq_hz"], ev_l2["freq_hz"], places=5)
        self.assertNotEqual(ev_l1["start_ms"], ev_l2["start_ms"])

    def test_causal_playhead_push_up_and_multiscale_decay(self) -> None:
        events = [
            {"kind": "tone", "freq_hz": 440.0, "t0": 20, "t1": 60, "gain": 0.9, "harmonics": 2},
            {"kind": "sweep", "freq_hz": 330.0, "end_freq_hz": 1320.0, "t0": 100, "t1": 180, "gain": 0.85},
        ]
        with ElectricAudioAgentSession(default_grid=(256, 64)) as session:
            # 1. Unplayed resting state (playhead_slot = -1): 0 bytes pushed up
            reg_rest = session.register_chargeable_audio(
                "src_play_test",
                "v1",
                events,
                playhead_slot=-1,
            )
            insp_rest = session.inspect_audio("src_play_test", "v1")
            cs_rest = insp_rest["charge_metrics"]["op_charge_summary_u8"]
            self.assertGreater(cs_rest["potential_byte_sum"], 0)
            self.assertEqual(cs_rest["pushed_up_byte_sum"], 0)
            self.assertEqual(cs_rest["pushed_up_active_bytes"], 0)

            # 2. Mid-play at slot 45 (inside first tone): bytes get charged up / pushed up!
            session.register_chargeable_audio(
                "src_play_test",
                "v2",
                events,
                playhead_slot=45,
            )
            insp_mid = session.inspect_audio("src_play_test", "v2", issue_observation_receipt=True)
            self.assertTrue(insp_mid["render"]["observed"])
            self.assertIsNotNone(insp_mid["observation_receipt"])
            cs_mid = insp_mid["charge_metrics"]["op_charge_summary_u8"]
            self.assertGreater(cs_mid["pushed_up_active_bytes"], 0)
            self.assertGreater(cs_mid["fast_charge_byte_sum"], 0)
            self.assertGreater(cs_mid["slow_charge_byte_sum"], 0)

            # 3. Late-play at slot 150 (inside sweep): first tone has decayed into slow afterglow
            session.register_chargeable_audio(
                "src_play_test",
                "v3",
                events,
                playhead_slot=150,
            )
            diff_rep = session.compare_audio_states(("src_play_test", "v2"), ("src_play_test", "v3"))
            self.assertGreater(diff_rep["charge_diff_metrics"]["pushed_up_added_bytes"], 0)
            self.assertGreater(diff_rep["charge_diff_metrics"]["discharged_removed_bytes"], 0)
            self.assertGreater(diff_rep["ahash64_hamming_distance"], 0)

            # Verify zeroization eviction
            ev = session.evict_chargeable_audio("src_play_test", "v1")
            self.assertTrue(ev["evicted"])
            self.assertEqual(ev["zeroized_bytes"], 256 * 64 * 5)
            assert_no_raw_buffer_leakage(insp_mid)
            assert_no_raw_buffer_leakage(diff_rep)

    def test_refractory_and_saturation_guard_against_loud_burst(self) -> None:
        # Sustained max-gain burst on the same frequency band across 30 time slots
        burst_events = [
            {"kind": "pulse", "freq_hz": 880.0, "t0": 10, "t1": 40, "gain": 1.0, "bandwidth_bands": 2}
        ]
        with ElectricAudioAgentSession(default_grid=(128, 32)) as session:
            session.register_chargeable_audio(
                "src_burst",
                "v1",
                burst_events,
                time_slots=128,
                freq_bands=32,
                playhead_slot=40,
                refractory_ticks=2,
            )
            insp = session.inspect_audio("src_burst", "v1")
            ref_guard = insp["charge_metrics"]["op_refractory_and_saturation_guard"]
            self.assertTrue(ref_guard["runaway_charge_prevented"])
            self.assertGreater(ref_guard["saturated_cell_count"], 0)
            self.assertGreater(ref_guard["refractory_clamped_count"], 0)
            self.assertLessEqual(
                insp["charge_metrics"]["op_push_up_profile_q16"]["max_charge_u8"],
                255,
            )

    def test_origin_attestation_and_semantic_refusal(self) -> None:
        events = [{"kind": "tone", "freq_hz": 440.0, "t0": 10, "t1": 30, "gain": 0.7}]
        with ElectricAudioAgentSession(default_grid=(128, 32)) as session:
            # owner_supplied_audio is accepted
            ok_reg = session.register_chargeable_audio(
                "src_supplied_aud",
                "v1",
                events,
                time_slots=128,
                freq_bands=32,
                origin_attestation="owner_supplied_audio",
            )
            self.assertEqual(ok_reg["origin_attestation"], "owner_supplied_audio")

            # Live microphone capture is deterministically refused
            for bad_origin in ("live_microphone_capture", "system_loopback_stream", "untrusted_stream"):
                with self.assertRaisesRegex(PermissionError, "ERR_NON_SYNTHETIC_AUDIO_PROHIBITED"):
                    session.register_chargeable_audio(
                        "src_supplied_aud",
                        "v2",
                        events,
                        time_slots=128,
                        freq_bands=32,
                        origin_attestation=bad_origin,
                    )

            # Semantic / ASR / emotional queries are deterministically refused
            ref = session.evaluate_agent_audio_query(
                "semantic_meaning",
                {"query_type": "semantic_meaning", "question": "Transcribe the spoken words and mood"},
                source_id="src_supplied_aud",
                source_version="v1",
            )
            self.assertEqual(ref["status"], "refused")
            self.assertEqual(ref["refusal_code"], REFUSAL_CODE_SEMANTIC)

    def test_continuous_electric_ear_stream_and_ghost_realization(self) -> None:
        spec = {"tone_text": "ACE"}
        with ElectricAudioAgentSession(default_grid=(256, 64)) as session:
            stream = session.create_continuous_ear_stream(
                "src_cont_ear",
                time_slots=256,
                freq_bands=64,
                max_window_frames=3,
                settle_ticks=1,
            )

            # Tick 1 (v1 at playhead_ratio=0.25): Verified with receipt
            t1 = stream.ingest_audio_tick("v1", spec, playhead_ratio=0.25, issue_observation_receipt=True)
            self.assertTrue(t1["observation_state"]["observed"])
            self.assertTrue(stream.check_receipt_freshness("v1")["fresh"])

            # Stage ghost prediction for playhead_ratio=0.6 (slot 153)
            staged = stream.stage_ghost_audio_prediction(
                "v1",
                [],
                ghost_playhead_slot=int(round(0.6 * 255)),
                ghost_id="gh_ear_v2",
            )
            self.assertEqual(staged["layer_isolation"], "ghost_isolation_plane")
            self.assertFalse(staged["observed"])

            # Tick 2 (v2 at playhead_ratio=0.6): v1 decays to STALE, ghost realization verified!
            t2 = stream.ingest_audio_tick(
                "v2",
                spec,
                playhead_ratio=0.6,
                issue_observation_receipt=True,
                verify_staged_ghost_id="gh_ear_v2",
            )
            self.assertIn("v1", t2["decayed_to_stale_versions"])
            self.assertEqual(stream.check_receipt_freshness("v1")["uncertainty_state"], "stale")
            self.assertTrue(t2["ghost_realization_audit"]["exact_realization"])
            self.assertEqual(t2["ghost_realization_audit"]["realization_convergence_score"], 1.0)

            # Tick 3 (v3 at playhead_ratio=1.0)
            t3 = stream.ingest_audio_tick("v3", spec, playhead_ratio=1.0)
            self.assertIn("v2", t3["decayed_to_stale_versions"])

            # Tick 4 (v4 at playhead_ratio=1.0): settles stream & evicts v1 with explicit_bzero!
            t4 = stream.ingest_audio_tick("v4", spec, playhead_ratio=1.0, issue_observation_receipt=True)
            self.assertIsNotNone(t4["ring_buffer_eviction"])
            self.assertTrue(t4["ring_buffer_eviction"]["evicted"])
            self.assertEqual(t4["ring_buffer_eviction"]["evicted_source"]["source_version"], "v1")
            self.assertEqual(t4["ring_buffer_eviction"]["zeroized_bytes"], 256 * 64 * 5)
            det4 = t4["temporal_window_summary"]["settling_and_oscillation_detector"]
            self.assertEqual(det4["regime"], "settled")
            self.assertTrue(det4["settled"])

    def test_multimodal_sight_and_sound_and_charge_packet_electrical_area(self) -> None:
        # 1. Verify 16-byte binary ChargePacket roundtrip
        pkt = build_charge_packet(
            t=340,
            modality=MODALITY_FUSED,
            f_band=72,
            q_fast=195,
            q_slow=110,
            revision=1,
        )
        raw16 = pack_charge_packet_16bytes(pkt)
        self.assertEqual(len(raw16), 16)
        unpacked = unpack_charge_packet_16bytes(raw16)
        self.assertEqual(unpacked, pkt)

        # 2. Verify Tests A, B, C, D + Step 5 Two-Lane Ablation
        rt_plan = run_realtime_fusion_test_plan()
        self.assertTrue(rt_plan["all_passed"])
        self.assertTrue(rt_plan["test_a_synchronization_100ms_passed"])
        self.assertTrue(rt_plan["test_b_unimodal_and_low_charge_passed"])
        self.assertTrue(rt_plan["test_c_revision_token_passed"])
        self.assertTrue(rt_plan["test_d_dedup_and_window_eviction_passed"])
        self.assertTrue(rt_plan["step_5_two_lane_ablation_passed"])

        # 3. Verify Multimodal Sight + Sound session (agents see and hear together)
        with ElectricSightAndSoundSession() as av_session:
            rep = av_session.inspect_sight_and_sound(
                "src_av_sight",
                "src_av_sound",
                "v1",
                audio_events_or_spec={"tone_text": "CYCLOPS"},
                playhead_ratio=0.8,
                issue_observation_receipts=True,
            )
            self.assertEqual(rep["multimodal_fusion"]["modal_alignment"], "audiovisual_coincident_active")
            self.assertTrue(rep["multimodal_fusion"]["projected_pushed_up_bytes_to_vision"])
            self.assertGreater(rep["multimodal_fusion"]["joint_salience_q16"], 0)
            self.assertTrue(rep["sight"]["render"]["observed"])
            self.assertTrue(rep["sound"]["render"]["observed"])

    def test_electric_audio_cli_conventions(self) -> None:
        code_st, out_st = run_cli(["--no-color", "--plain", "selftest"])
        self.assertEqual(code_st, 0)
        self.assertNotIn("\033[", out_st)
        self.assertTrue(out_st.isascii())
        self.assertIn("PASS", out_st)

        code_insp, out_insp = run_cli([
            "--no-color",
            "--plain",
            "--mode",
            "focus",
            "inspect-audio",
            "--tone-text",
            "HELLO",
            "--playhead-ratio",
            "0.75",
            "--issue-receipt",
        ])
        self.assertEqual(code_insp, 0)
        self.assertNotIn("\033[", out_insp)
        self.assertTrue(out_insp.isascii())
        self.assertIn("OBS:VERIFIED_RECEIPT", out_insp)

        code_av, out_av = run_cli([
            "--json",
            "sight-and-sound",
            "--tone-text",
            "HI",
            "--issue-receipts",
        ])
        self.assertEqual(code_av, 0)
        av_data = json.loads(out_av)
        self.assertEqual(av_data["multimodal_fusion"]["modal_alignment"], "audiovisual_coincident_active")

        code_tr, out_tr = run_cli([
            "--no-color",
            "--plain",
            "charge-trace",
            "--tone-text",
            "HI",
            "--top-k",
            "3",
        ])
        self.assertEqual(code_tr, 0)
        self.assertNotIn("\033[", out_tr)
        self.assertTrue(out_tr.isascii())
        self.assertIn("Explainable Charge Push-Up Trace", out_tr)

    def test_feedback_remediation_a_to_e_tier2_pcm_narration_and_trace(self) -> None:
        # Point A: Semantic Boundary Contract & non-authoritative AgentInterpretationEnvelope
        self.assertEqual(
            SEMANTIC_BOUNDARY_CONTRACT["enclave_sensor_layer"]["is_enclave_truth"],
            True,
        )
        self.assertEqual(
            SEMANTIC_BOUNDARY_CONTRACT["agent_interpreter_layer"]["is_enclave_truth"],
            False,
        )
        interp = create_agent_interpretation_envelope(
            enclave_receipt_id="rcpt_test_01",
            enclave_replay_digest="sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            model_id="cyclops-narrator-v1",
            prompt_version="p1",
            interpreted_tokens=["person_clapped", "quiet_room"],
            confidence_q16=55050,
        )
        self.assertFalse(interp["is_enclave_truth"])
        self.assertFalse(interp["may_cite_as_enclave_truth"])
        self.assertEqual(interp["epistemic_origin"], "agent_probabilistic_interpreter")

        # Point B: Consented Tier-2 PCM validator refuses missing/invalid consent_token
        with self.assertRaisesRegex(PermissionError, "ERR_MISSING_AUDIO_CONSENT_TOKEN"):
            validate_and_translate_consented_pcm_samples(
                [1000, -1000] * 100,
                sample_rate_hz=16000,
                origin_attestation="owner_supplied_audio",
                consent_token=None,
            )

        # Point C: Explainable charge trace & adaptive threshold calibration
        trace = explain_charge_push_up_trace(
            [{"kind": "tone", "freq_hz": 440.0, "t0": 10, "t1": 40, "gain": 0.9}],
            time_slots=128,
            freq_bands=32,
            playhead_slot=35,
        )
        self.assertGreater(trace["pushed_up_slot_count"], 0)
        self.assertGreater(len(trace["explainable_trace"]), 0)

        # Point D: AgentNarrationStateMachine coalescing & revision amendment policy
        narr = AgentNarrationStateMachine(coalesce_window_ms=300, min_revision_confidence_delta_q16=8192)
        self.assertIsNone(
            narr.push_packet(build_charge_packet(t=100, modality=MODALITY_VISION, f_band=24, q_fast=130, q_slow=60))
        )
        utt1 = narr.push_packet(
            build_charge_packet(t=420, modality=MODALITY_VISION, f_band=26, q_fast=140, q_slow=70)
        )
        self.assertIsNotNone(utt1)
        self.assertFalse(utt1["is_revision"])
        narr.push_packet(
            build_charge_packet(t=500, modality=MODALITY_FUSED, f_band=96, q_fast=240, q_slow=180, revision=1)
        )
        utt2 = narr.flush(current_t_ms=800)
        self.assertIsNotNone(utt2)
        self.assertTrue(utt2["is_revision"])
        self.assertEqual(utt2["amends_utterance_id"], utt1["utterance_id"])

        # Point E: Run comprehensive Tier-2 fidelity, adversarial torture & edge latency suite
        suite = run_tier2_fidelity_adversarial_and_edge_profile()
        self.assertTrue(suite["all_passed"])
        self.assertTrue(suite["speech_vs_bass_discrimination_passed"])
        self.assertTrue(suite["adversarial_sanitization_passed"])
        self.assertTrue(suite["edge_latency_target_under_50ms_passed"])

        # Verify 4-band polyphonic decomposition & dynamic drop profile on synthetic drop signal
        drop_pcm = [200] * 2000 + [
            int(12000 * ((i % 16) / 8.0 - 1.0)) for i in range(6000)
        ]
        drop_res = validate_and_translate_consented_pcm_samples(
            drop_pcm,
            sample_rate_hz=16000,
            origin_attestation="owner_supplied_audio",
            consent_token={"consented": True, "token_id": "tok_drop"},
            time_slots=128,
            freq_bands=32,
        )
        self.assertIn("four_band_polyphonic_profile", drop_res)
        self.assertIn("dynamic_and_tempo_profile", drop_res)
        self.assertTrue(drop_res["dynamic_and_tempo_profile"]["section_drop_detected"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
