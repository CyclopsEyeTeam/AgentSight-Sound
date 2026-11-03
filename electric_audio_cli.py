#!/usr/bin/env python3
"""
electric_audio_cli.py (cyclops-ea-ctl)

Standalone CLI and Terminal Companion Formatter for the Chargeable-Byte Electric
Audio ("Electric Ear") and Multimodal Sight+Sound stack.

Adheres strictly to Prism Terminal Companion & Electric Eye Governance Conventions:
- Supports `--once` for lightweight snapshotting and `--mode focus` for full telemetry.
- Supports `NO_COLOR` environment variable, `--no-color`, and `--plain` ASCII
  rendering without emitting broken ANSI escape sequences.
- Zero raw waveform buffer, chargeable-byte buffer, or cryptographic key leakage.
- Enforces explicit epistemic badges (`[OBS:VERIFIED_RECEIPT]`, `[UNOBSERVED_RENDER]`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

from electric_audio_agent_bridge import (
    SEMANTIC_BOUNDARY_CONTRACT,
    SUPPORTED_AUDIO_GRIDS,
    ElectricAudioAgentSession,
    ElectricSightAndSoundSession,
    calibrate_adaptive_charge_thresholds,
    explain_charge_push_up_trace,
    run_realtime_fusion_test_plan,
    run_tier2_fidelity_adversarial_and_edge_profile,
)
from electric_eye_harness import assert_no_raw_buffer_leakage
from electric_vision_cli import (
    _load_marks_arg,
    _resolve_color_and_plain,
    format_terminal_banner,
)


def _load_audio_spec_arg(
    events_spec: Optional[str],
    tone_text: Optional[str] = None,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Loads an audio specification from `--tone-text`, inline JSON, or a JSON file path."""
    if tone_text is not None and tone_text.strip():
        return {"tone_text": tone_text}
    if not events_spec:
        raise ValueError("Either --events or --tone-text must be provided")
    s = events_spec.strip()
    if s.startswith("[") or s.startswith("{"):
        parsed = json.loads(s)
    elif os.path.isfile(s):
        with open(s, "r", encoding="utf-8") as f:
            parsed = json.load(f)
    else:
        raise ValueError(
            f"Expected inline JSON or existing file path for audio events, got: {events_spec!r}"
        )
    if isinstance(parsed, (list, dict)):
        return parsed
    raise ValueError("Audio events JSON must be a list of event objects or a spec dict")


def render_audio_inspect_terminal_output(
    report: Dict[str, Any],
    *,
    mode: str = "once",
    use_color: bool = True,
    use_plain: bool = False,
) -> str:
    """Renders an `inspect_audio` report for terminal output without any ANSI leakage when disabled."""
    rnd = report["render"]
    state = rnd["uncertainty_state"]
    observed = bool(rnd["observed"])
    grid = report["audio_grid"]
    cm = report["charge_metrics"]
    csum = cm["op_charge_summary_u8"]
    pprof = cm["op_push_up_profile_q16"]
    spec = cm["op_spectral_centroid_and_spread_q16"]
    ms = cm["op_multiscale_fast_slow_ratio_q16"]
    ref = cm["op_refractory_and_saturation_guard"]
    cc = cm["op_connected_charge_events_4c"]
    ph = cm["op_playhead_phase_profile"]

    lines: List[str] = [
        format_terminal_banner(
            state,
            f"Electric Audio Inspect {report['source_id']}@{report['source_version']} ({grid['time_slots']}x{grid['freq_bands']} chargeable bytes)",
            observed=observed,
            use_color=use_color,
            use_plain=use_plain,
        ),
        f"  Render ID          : {rnd['render_id']}",
        f"  Layer Isolation    : {rnd['layer_isolation']}",
        f"  Observed Status    : {observed} (Receipt: {rnd.get('observation_receipt_id') or 'NONE'})",
        f"  Playhead Slot      : {ph['playhead_slot']} / {ph['time_slots'] - 1}",
        f"  Pushed-Up Bytes    : {csum['pushed_up_active_bytes']} / {csum['total_chargeable_bytes']} (sum={csum['pushed_up_byte_sum']}, peak={pprof['max_charge_u8']})",
        f"  Multi-Scale Charge : fast_sum={csum['fast_charge_byte_sum']}, slow_sum={csum['slow_charge_byte_sum']} ({ms['regime']})",
        f"  Spectral Centroid  : {spec['nominal_centroid_hz']} Hz (band_q16={spec['centroid_band_q16']})",
        f"  Event Islands (4C) : {cc['component_count']} (bbox={cc['foreground_bbox']})",
        f"  Refractory Guard   : saturated={ref['saturated_cell_count']}, clamped={ref['refractory_clamped_count']}",
        f"  Perceptual aHash64 : {cm['op_perceptual_audio_hash_ahash64']}",
        f"  Replay Digest      : {report['audio_replay_digest']}",
    ]

    if mode == "focus":
        lines.append(f"  Charge 16-Bin Hist : {pprof['charge_histogram_16bin']}")
        lines.append(f"  Envelope 16-Seg u8 : {ph['reconstructible_envelope_16seg_u8']}")
        lines.append(
            f"  Playhead Phases    : pre_uncharged={ph['pre_play_uncharged_bytes']}, "
            f"active_win={ph['active_playhead_window_bytes']}, afterglow={ph['post_play_afterglow_bytes']}"
        )
        aria_str = str(report["accessibility"]["aria_label"])
        if use_plain:
            aria_str = aria_str.encode("ascii", errors="replace").decode("ascii")
        lines.append(f"  ARIA Label         : {aria_str}")
    return "\n".join(lines)


def render_audio_stream_terminal_output(
    stream_report: Dict[str, Any],
    *,
    mode: str = "once",
    use_color: bool = True,
    use_plain: bool = False,
) -> str:
    """Renders a `ContinuousElectricEarStream` playback report for terminal output."""
    summary = stream_report["final_window_summary"]
    grid = stream_report["audio_grid"]
    det = summary["settling_and_oscillation_detector"]
    ahash_tl = summary["ahash64_hamming_timeline"]
    watchdog = summary["epistemic_watchdog"]
    kin = summary["pitch_kinematic_trajectory_chain"]
    dirty_prof = summary["dirty_charge_tile_16x16_profile"]

    banner_state = "verified" if watchdog["fresh_verified_versions"] else "derived"
    observed = bool(watchdog["fresh_verified_versions"])
    lines: List[str] = [
        format_terminal_banner(
            banner_state,
            f"Continuous Electric Ear Stream {stream_report['source_id']} ({grid['time_slots']}x{grid['freq_bands']})",
            observed=observed,
            use_color=use_color,
            use_plain=use_plain,
        ),
        f"  Ticks Ingested     : {stream_report['total_ticks_ingested']} (Window Active: {summary['active_frame_count']}/{summary['window_capacity']})",
        f"  Evicted & Zeroized : {summary['evicted_frame_count']} frames ({summary['cumulative_zeroized_bytes']} bytes explicit_bzero)",
        f"  Stream Regime      : {det['regime']} (settled={det['settled']}, cycle={det['oscillating_cycle_detected']})",
        f"  aHash64 Hamming    : steps={ahash_tl['step_hamming_distances']} (max={ahash_tl['max_step_hamming']}, checkpoints={ahash_tl['checkpoints_triggered']})",
        f"  Dirty 16x16 Tiles  : mean_ratio={dirty_prof['mean_dirty_tile_ratio']}",
        f"  Fresh Verified     : {watchdog['fresh_verified_versions']}",
        f"  Decayed [STALE]    : {watchdog['decayed_stale_versions']}",
    ]
    if mode == "focus":
        if watchdog["decayed_stale_versions"]:
            stale_banner = format_terminal_banner(
                "stale",
                f"Epistemic Watchdog: {len(watchdog['decayed_stale_versions'])} Prior Audio Observation(s) Superseded -> STALE",
                observed=False,
                use_color=use_color,
                use_plain=use_plain,
            )
            lines.append(stale_banner)
        lines.append(f"  Pitch Velocities   : {kin['velocities']}")
        lines.append(f"  Next Ghost Pitch   : {kin['predicted_next_centroid']}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cyclops-ea-ctl",
        description="Electric Audio & Multimodal Sight+Sound CLI for Agents (Chargeable-Byte Enclave)",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color codes")
    parser.add_argument("--plain", action="store_true", help="Use plain 7-bit ASCII borders and glyphs")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON output")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single lightweight snapshot and exit immediately (default behavior)",
    )
    parser.add_argument(
        "--mode",
        choices=("once", "focus"),
        default="once",
        help="Telemetry verbosity mode ('once' for compact snapshot, 'focus' for full telemetry)",
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    # 1. status
    subparsers.add_parser("status", help="Snapshot Electric Audio & Vision enclave status")

    # 2. selftest
    subparsers.add_parser("selftest", help="Run Chargeable-Byte Audio & Sight+Sound self-test")

    # 3. inspect-audio
    p_inspect = subparsers.add_parser(
        "inspect-audio",
        help="Translate audio events or Tone Alphabet text to chargeable bytes, push up to playhead, and inspect",
    )
    p_inspect.add_argument("--events", help="Inline JSON array/dict of audio events or path to JSON file")
    p_inspect.add_argument("--tone-text", help="Tone Alphabet V0 ASCII text (e.g. 'HELLO CYCLOPS')")
    p_inspect.add_argument("--time-slots", type=int, default=256, help="Time slots (128, 256, or 512)")
    p_inspect.add_argument("--freq-bands", type=int, default=64, help="Frequency bands (32, 64, or 128)")
    p_inspect.add_argument(
        "--playhead-ratio",
        type=float,
        default=1.0,
        help="Normalized playhead position in [0.0..1.0], or -1.0 for unplayed resting state",
    )
    p_inspect.add_argument(
        "--issue-receipt",
        action="store_true",
        help="Mint capability token and issue HSM-countersigned ObservationReceipt",
    )

    # 4. diff-audio
    p_diff = subparsers.add_parser(
        "diff-audio",
        help="Compare chargeable-byte push-up states between two audio specifications or playhead positions",
    )
    p_diff.add_argument("--events-a", required=True, help="Before audio events (inline JSON or file)")
    p_diff.add_argument("--events-b", required=True, help="After audio events (inline JSON or file)")
    p_diff.add_argument("--playhead-ratio-a", type=float, default=0.5, help="Playhead ratio for state A")
    p_diff.add_argument("--playhead-ratio-b", type=float, default=1.0, help="Playhead ratio for state B")
    p_diff.add_argument("--time-slots", type=int, default=256)
    p_diff.add_argument("--freq-bands", type=int, default=64)

    # 5. preview-ghost
    p_ghost = subparsers.add_parser(
        "preview-ghost",
        help="Project proposed audio events onto ghost_isolation_plane without mutating verified_plane",
    )
    p_ghost.add_argument("--base-events", required=True, help="Verified base audio events")
    p_ghost.add_argument("--ghost-events", required=True, help="Proposed ghost audio events")
    p_ghost.add_argument("--time-slots", type=int, default=256)
    p_ghost.add_argument("--freq-bands", type=int, default=64)

    # 6. refusal-check
    p_refuse = subparsers.add_parser(
        "refusal-check",
        help="Verify deterministic epistemic refusal on semantic speech/ASR, emotion, or intent queries",
    )
    p_refuse.add_argument("--query", required=True, help="Semantic/ASR/emotional question to test")
    p_refuse.add_argument("--query-type", default="semantic_meaning", help="Query type")

    # 7. play-stream
    p_stream = subparsers.add_parser(
        "play-stream",
        help="Stream playhead across chargeable bytes so bytes get charged up / pushed up tick by tick",
    )
    p_stream.add_argument("--events", help="Inline JSON array/dict of audio events or path to JSON file")
    p_stream.add_argument("--tone-text", help="Tone Alphabet V0 ASCII text (e.g. 'HELLO')")
    p_stream.add_argument(
        "--playhead-ratios",
        default="[0.25, 0.5, 0.75, 1.0, 1.0]",
        help="JSON array of playhead ratios [0.0..1.0] to step through",
    )
    p_stream.add_argument("--time-slots", type=int, default=256)
    p_stream.add_argument("--freq-bands", type=int, default=64)
    p_stream.add_argument("--window-size", type=int, default=4)
    p_stream.add_argument("--settle-ticks", type=int, default=1)
    p_stream.add_argument("--issue-receipts", action="store_true")

    # 8. sight-and-sound
    p_av = subparsers.add_parser(
        "sight-and-sound",
        help="Run multimodal Electric Sight + Electric Audio session (agents see and hear together)",
    )
    p_av.add_argument("--events", help="Inline JSON audio events or file path")
    p_av.add_argument("--tone-text", help="Tone Alphabet V0 ASCII text")
    p_av.add_argument(
        "--marks",
        help="Optional visual marks JSON; if omitted, pushed-up audio bytes are projected onto the visual canvas",
    )
    p_av.add_argument("--playhead-ratio", type=float, default=1.0)
    p_av.add_argument("--issue-receipts", action="store_true")

    # 9. charge-trace
    p_trace = subparsers.add_parser(
        "charge-trace",
        help="Inspect explainable causal push-up trace and adaptive session threshold calibration",
    )
    p_trace.add_argument("--events", help="Inline JSON audio events or file path")
    p_trace.add_argument("--tone-text", help="Tone Alphabet V0 ASCII text")
    p_trace.add_argument("--playhead-ratio", type=float, default=1.0)
    p_trace.add_argument("--top-k", type=int, default=6)

    return parser


def run_cli(argv: Optional[Sequence[str]] = None) -> tuple[int, str]:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    use_color, use_plain = _resolve_color_and_plain(args.no_color, args.plain)
    subcmd = args.subcommand or "status"

    if subcmd == "status":
        with ElectricAudioAgentSession(default_grid=(256, 64)) as session:
            health = session.vision_session.get_enclave_health_and_audit_summary()
        payload = {
            "command": "status",
            "mode": args.mode,
            "supported_audio_grids": [f"{t}x{b}" for t, b in SUPPORTED_AUDIO_GRIDS],
            "enclave_health": health,
        }
        assert_no_raw_buffer_leakage(payload)
        if args.json:
            return 0, json.dumps(payload, indent=2, sort_keys=True)
        banner = format_terminal_banner(
            "verified",
            "Electric Audio (Chargeable-Byte Enclave) Status",
            observed=False,
            use_color=use_color,
            use_plain=use_plain,
        )
        lines = [
            banner,
            f"  SIMD Backend       : {health['native_simd_backend']}",
            f"  HSM Key ID         : {health['hsm_enclave'].get('active_key_id')}",
            f"  Audio Charge Grids : {', '.join(payload['supported_audio_grids'])}",
            f"  Kill Switches Trig : {health['quarantine_supervisor']['triggered_kill_switches']}",
        ]
        if args.mode == "focus":
            lines.append(f"  Audit Chain Head   : {health['audit_worm_ledger'].get('chain_head_digest')}")
        return 0, "\n".join(lines)

    if subcmd == "selftest":
        with ElectricSightAndSoundSession() as av_session:
            # Verify resting unplayed (0 pushed-up bytes) vs played (pushed-up bytes > 0)
            av_session.audio.register_chargeable_audio(
                "src_ea_selftest",
                "v1",
                {"tone_text": "HI"},
                playhead_slot=-1,
            )
            rest_rep = av_session.audio.inspect_audio("src_ea_selftest", "v1")
            av_rep = av_session.inspect_sight_and_sound(
                "src_ev_selftest",
                "src_ea_selftest",
                "v2",
                audio_events_or_spec={"tone_text": "HI"},
                playhead_ratio=1.0,
                issue_observation_receipts=True,
            )

        rest_pushed = rest_rep["charge_metrics"]["op_charge_summary_u8"]["pushed_up_active_bytes"]
        play_pushed = av_rep["sound"]["charge_metrics"]["op_charge_summary_u8"]["pushed_up_active_bytes"]
        sight_objs = av_rep["sight"]["native_simd_metrics"]["op_connected_components_4c"]["component_count"]
        rt_plan = run_realtime_fusion_test_plan()
        tier2_prof = run_tier2_fidelity_adversarial_and_edge_profile()
        all_ok = bool(
            rest_pushed == 0
            and play_pushed > 0
            and sight_objs > 0
            and av_rep["sound"]["render"]["observed"] is True
            and av_rep["sight"]["render"]["observed"] is True
            and rt_plan["all_passed"] is True
            and tier2_prof["all_passed"] is True
        )
        payload = {
            "command": "selftest",
            "all_passed": all_ok,
            "resting_unplayed_pushed_up_bytes": rest_pushed,
            "playing_pushed_up_bytes": play_pushed,
            "projected_sight_components": sight_objs,
            "joint_salience_q16": av_rep["multimodal_fusion"]["joint_salience_q16"],
            "realtime_fusion_tests_a_b_c_d_ablation": rt_plan,
            "tier2_fidelity_adversarial_and_edge_profile": tier2_prof,
            "semantic_boundary_contract_id": SEMANTIC_BOUNDARY_CONTRACT["contract_id"],
            "audio_replay_digest": av_rep["sound"]["audio_replay_digest"],
        }
        assert_no_raw_buffer_leakage(payload)
        if args.json:
            return (0 if all_ok else 1), json.dumps(payload, indent=2, sort_keys=True)
        banner = format_terminal_banner(
            "verified" if all_ok else "stale",
            f"Electric Audio & Sight+Sound Self-Test: {'PASS' if all_ok else 'FAIL'}",
            observed=True,
            use_color=use_color,
            use_plain=use_plain,
        )
        lines = [
            banner,
            f"  Unplayed Resting   : {rest_pushed} pushed-up bytes (PASS)",
            f"  Played Charged-Up  : {play_pushed} pushed-up bytes (PASS)",
            f"  Sight Projection   : {sight_objs} connected visual components (PASS)",
            f"  Tests A/B/C/D+Abl  : {'PASS' if rt_plan['all_passed'] else 'FAIL'} (100ms align_id + Electrical Area)",
            f"  Tier-2 + Adv + Edge: {'PASS' if tier2_prof['all_passed'] else 'FAIL'} (batch={tier2_prof['edge_profile']['batch_200_packets_elapsed_ms']}ms)",
            f"  Boundary Contract  : {SEMANTIC_BOUNDARY_CONTRACT['contract_id']}",
            f"  Joint Salience Q16 : {payload['joint_salience_q16']}",
            f"  Audio Replay Digest: {payload['audio_replay_digest']}",
        ]
        return (0 if all_ok else 1), "\n".join(lines)

    if subcmd == "inspect-audio":
        spec = _load_audio_spec_arg(args.events, args.tone_text)
        with ElectricAudioAgentSession(default_grid=(args.time_slots, args.freq_bands)) as session:
            session.register_chargeable_audio(
                "src_cli_audio",
                "v1",
                spec,
                time_slots=args.time_slots,
                freq_bands=args.freq_bands,
                playhead_ratio=args.playhead_ratio,
            )
            rep = session.inspect_audio(
                "src_cli_audio",
                "v1",
                issue_observation_receipt=args.issue_receipt,
            )
        assert_no_raw_buffer_leakage(rep)
        if args.json:
            return 0, json.dumps(rep, indent=2, sort_keys=True)
        return 0, render_audio_inspect_terminal_output(
            rep, mode=args.mode, use_color=use_color, use_plain=use_plain
        )

    if subcmd == "diff-audio":
        spec_a = _load_audio_spec_arg(args.events_a)
        spec_b = _load_audio_spec_arg(args.events_b)
        with ElectricAudioAgentSession(default_grid=(args.time_slots, args.freq_bands)) as session:
            session.register_chargeable_audio(
                "src_cli_adiff",
                "v1",
                spec_a,
                time_slots=args.time_slots,
                freq_bands=args.freq_bands,
                playhead_ratio=args.playhead_ratio_a,
            )
            session.register_chargeable_audio(
                "src_cli_adiff",
                "v2",
                spec_b,
                time_slots=args.time_slots,
                freq_bands=args.freq_bands,
                playhead_ratio=args.playhead_ratio_b,
            )
            rep = session.compare_audio_states(("src_cli_adiff", "v1"), ("src_cli_adiff", "v2"))
        assert_no_raw_buffer_leakage(rep)
        if args.json:
            return 0, json.dumps(rep, indent=2, sort_keys=True)
        diff = rep["charge_diff_metrics"]
        banner = format_terminal_banner(
            "derived",
            f"Electric Audio Charge Diff src_cli_adiff@v1 -> v2 ({args.time_slots}x{args.freq_bands})",
            observed=False,
            use_color=use_color,
            use_plain=use_plain,
        )
        lines = [
            banner,
            f"  Changed Bytes      : {diff['changed_bytes']} / {diff['total_bytes']} (ratio={diff['change_ratio']})",
            f"  Pushed-Up / Dischg : +{diff['pushed_up_added_bytes']} / -{diff['discharged_removed_bytes']}",
            f"  Net Charge Delta   : {rep['net_push_up_byte_delta']}",
            f"  aHash64 Hamming    : {rep['ahash64_hamming_distance']}",
            f"  Diff Replay Digest : {rep['diff_replay_digest']}",
        ]
        return 0, "\n".join(lines)

    if subcmd == "preview-ghost":
        base_spec = _load_audio_spec_arg(args.base_events)
        ghost_spec = _load_audio_spec_arg(args.ghost_events)
        with ElectricAudioAgentSession(default_grid=(args.time_slots, args.freq_bands)) as session:
            session.register_chargeable_audio(
                "src_cli_aghost",
                "v1",
                base_spec,
                time_slots=args.time_slots,
                freq_bands=args.freq_bands,
                playhead_ratio=1.0,
            )
            rep = session.preview_ghost_audio("src_cli_aghost", "v1", ghost_spec)
        assert_no_raw_buffer_leakage(rep)
        if args.json:
            return 0, json.dumps(rep, indent=2, sort_keys=True)
        banner = format_terminal_banner(
            "predicted",
            f"GhostOverlay Audio Projection on src_cli_aghost@v1 ({args.time_slots}x{args.freq_bands})",
            observed=False,
            use_color=use_color,
            use_plain=use_plain,
        )
        hm = rep["hypothetical_metrics"]
        lines = [
            banner,
            f"  Layer Isolation    : {rep['layer_isolation']}",
            f"  Observed Status    : {rep['observed']} (Ghost Byte Leakage: {rep['ghost_byte_leakage']})",
            f"  Base SHA Preserved : {rep['base_charged_sha256_preserved']}",
            f"  Hypothetical Pushed: {hm['op_charge_summary_u8']['pushed_up_active_bytes']} bytes",
            f"  Hypothetical Diff  : {hm['op_bbox_aligned_charge_diff_u8']['changed_bytes']} bytes vs base",
        ]
        return 0, "\n".join(lines)

    if subcmd == "refusal-check":
        with ElectricAudioAgentSession(default_grid=(128, 32)) as session:
            session.register_chargeable_audio(
                "src_cli_arefuse",
                "v1",
                [{"kind": "tone", "freq_hz": 440.0, "start_ms": 0.0, "duration_ms": 120.0, "gain": 0.8}],
                time_slots=128,
                freq_bands=32,
            )
            rep = session.evaluate_agent_audio_query(
                args.query_type,
                {"query_type": args.query_type, "question": args.query},
                source_id="src_cli_arefuse",
                source_version="v1",
            )
        assert_no_raw_buffer_leakage(rep)
        if args.json:
            return 0, json.dumps(rep, indent=2, sort_keys=True)
        banner = format_terminal_banner(
            "unknown",
            f"Audio Semantic Refusal Gate: {rep['status'].upper()}",
            observed=False,
            use_color=use_color,
            use_plain=use_plain,
        )
        lines = [
            banner,
            f"  Refusal Code       : {rep.get('refusal_code')}",
            f"  Refusal Reason     : {rep.get('refusal_reason')}",
            f"  Result             : {rep.get('result')}",
        ]
        return 0, "\n".join(lines)

    if subcmd == "play-stream":
        spec = _load_audio_spec_arg(args.events, args.tone_text)
        ratios = json.loads(args.playhead_ratios)
        if not isinstance(ratios, list) or not ratios:
            raise ValueError("--playhead-ratios must be a non-empty JSON list of floats")

        tick_reports: List[Dict[str, Any]] = []
        with ElectricAudioAgentSession(default_grid=(args.time_slots, args.freq_bands)) as session:
            stream = session.create_continuous_ear_stream(
                "src_cli_ear_stream",
                time_slots=args.time_slots,
                freq_bands=args.freq_bands,
                max_window_frames=args.window_size,
                settle_ticks=args.settle_ticks,
            )
            for idx, r_val in enumerate(ratios):
                ver = f"v{idx + 1}"
                tick_rep = stream.ingest_audio_tick(
                    ver,
                    spec,
                    playhead_ratio=float(r_val),
                    issue_observation_receipt=args.issue_receipts,
                )
                tick_reports.append(tick_rep)
            final_summary = stream.compute_temporal_window_summary()

        payload = {
            "command": "play-stream",
            "source_id": "src_cli_ear_stream",
            "audio_grid": {"time_slots": args.time_slots, "freq_bands": args.freq_bands},
            "total_ticks_ingested": len(ratios),
            "ticks": tick_reports,
            "final_window_summary": final_summary,
        }
        assert_no_raw_buffer_leakage(payload)
        if args.json:
            return 0, json.dumps(payload, indent=2, sort_keys=True)
        return 0, render_audio_stream_terminal_output(
            payload, mode=args.mode, use_color=use_color, use_plain=use_plain
        )

    if subcmd == "sight-and-sound":
        spec = _load_audio_spec_arg(args.events, args.tone_text)
        v_marks = _load_marks_arg(args.marks) if args.marks else None
        with ElectricSightAndSoundSession() as av_session:
            rep = av_session.inspect_sight_and_sound(
                "src_cli_sight",
                "src_cli_sound",
                "v1",
                visual_marks=v_marks,
                audio_events_or_spec=spec,
                playhead_ratio=args.playhead_ratio,
                issue_observation_receipts=args.issue_receipts,
            )
        assert_no_raw_buffer_leakage(rep)
        if args.json:
            return 0, json.dumps(rep, indent=2, sort_keys=True)
        fusion = rep["multimodal_fusion"]
        sight_cc = rep["sight"]["native_simd_metrics"]["op_connected_components_4c"]
        sound_cs = rep["sound"]["charge_metrics"]["op_charge_summary_u8"]
        state = "verified" if args.issue_receipts else "derived"
        banner = format_terminal_banner(
            state,
            "Multimodal Electric Sight + Electric Sound Inspection",
            observed=bool(args.issue_receipts),
            use_color=use_color,
            use_plain=use_plain,
        )
        lines = [
            banner,
            f"  Modal Alignment    : {fusion['modal_alignment']}",
            f"  Audio Pushed Bytes : {sound_cs['pushed_up_active_bytes']} / {sound_cs['total_chargeable_bytes']} (sum={sound_cs['pushed_up_byte_sum']})",
            f"  Visual Components  : {sight_cc['component_count']} (Projected from Audio: {fusion['projected_pushed_up_bytes_to_vision']})",
            f"  Joint Salience Q16 : {fusion['joint_salience_q16']} (Sight={fusion['visual_activity_q16']}, Sound={fusion['audio_push_up_activity_q16']})",
            f"  Fusion Digest      : {fusion['fusion_replay_digest']}",
        ]
        return 0, "\n".join(lines)

    if subcmd == "charge-trace":
        spec = _load_audio_spec_arg(args.events, args.tone_text)
        playhead_slot = int(round(max(0.0, min(1.0, float(args.playhead_ratio))) * 255))
        trace = explain_charge_push_up_trace(
            spec,
            time_slots=256,
            freq_bands=64,
            playhead_slot=playhead_slot,
        )
        cal = trace["calibrated_thresholds"]
        payload = {
            "command": "charge-trace",
            "adaptive_calibration": cal,
            "explainable_trace": trace,
        }
        assert_no_raw_buffer_leakage(payload)
        if args.json:
            return 0, json.dumps(payload, indent=2, sort_keys=True)
        banner = format_terminal_banner(
            "derived",
            f"Explainable Charge Push-Up Trace src_cli_trace@v1 (top_k={args.top_k})",
            observed=False,
            use_color=use_color,
            use_plain=use_plain,
        )
        lines = [
            banner,
            f"  Adaptive Thresholds: fast_thr={cal['calibrated_push_threshold_fast']}, slow_thr={cal['calibrated_push_threshold_slow']}",
            f"  Pushed-Up Slots    : {trace['pushed_up_slot_count']}",
        ]
        for item in trace["explainable_trace"][: args.top_k]:
            lines.append(
                f"  - slot={item['t_slot']} band={item['freq_band']} ({item['nominal_hz']}Hz): "
                f"q={item['pushed_up_byte_u8']} (fast={item['current_fast_charge_u8']}, slow={item['current_slow_charge_u8']}) -> {', '.join(item['push_trigger_reasons'])}"
            )
        return 0, "\n".join(lines)

    return 1, "Unknown subcommand"


if __name__ == "__main__":
    code, output = run_cli()
    print(output)
    sys.exit(code)
