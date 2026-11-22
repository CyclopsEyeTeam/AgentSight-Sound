#!/usr/bin/env python3
"""
watch_together.py — One-Piece Combined Sight + Sound + Vibration Watch-Together Engine

Wires together in a single self-contained session (`WatchTogetherSession`):
  1. Continuous Electric Eye (`ContinuousElectricEyeStream`, 512x512, window=8,
     `explicit_bzero` eviction, 3x3 coarse light/colour/brightest + temporal motion/regime).
  2. Chargeable-Byte Electric Ear (`validate_and_translate_consented_pcm_samples` +
     `ElectricAudioAgentSession` + 10 ms onset tempo tracker + 4-band polyphonic profile).
  3. Horizontal Charge Spread (`apply_horizontal_spread`) + 8-Byte `VibrationPacket`
     (`>IHBB`: `t`, `align_id`, `q_vibe`, `h_spread`) + `HapticActuator` (6 safety gates).
  4. Real-Time `AgentElectricalAreaBuffer` (`align_id` 100 ms window fusion of
     Vision `modality=0` + Audio `modality=1` + Vibration into `modality=2` / `modality=3`,
     16-byte `ChargePacket` + 8-byte `VibrationPacket`) + `AgentNarrationStateMachine`
     (anti-chatter coalescing & revision tokens).
  5. Multi-Resident Watch-Party Lane Attention (`ResidentAttentionLane`) so everyone in
     Cyclops Chat (Companion, Claude/Spark, Gemini/Prism, Nemotron, Muse, etc.) watches
     the same shared moment from the one sensory bus while noticing what fits their own
     sensibility — strictly keeping Layer A (`is_enclave_truth=True`, receipts & digests)
     separate from Layer B (`is_enclave_truth=False`, chat reactions & diary notes).

Supports three zero-disk-leakage sources:
  - `watch_video_file(path, ...)`: watches any `.mp4`/`.mkv`/`.webm` video (or pairs a
    visual animation/video with an `.mp3`/`.wav` soundtrack) across synchronized windows.
  - `glance_and_listen_live(listen_seconds=2.0)`: simultaneous live desktop glance
    (`origin_attestation="companion_desktop_glance"`) + default sink monitor audio listen
    (`origin_attestation="companion_system_audio"`, never the microphone) fused into the
    same `align_id` window.
  - `watch_synthetic_cinema(...)`: zero-dependency self-contained audiovisual cinema
    generator for instant testing anywhere.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

import visual_sources
from desktop_ear import (
    ContinuousElectricEarStream,
    ContinuousSystemAudioMonitor,
    DesktopEar,
)
from desktop_sight import (
    ORIGIN as DESKTOP_GLANCE_ORIGIN,
    REGIONS,
    SIZE as DEFAULT_SIGHT_SIZE,
    _capture as _capture_root_desktop_image,
    _coarse as _measure_coarse_3x3,
    _movement as _describe_movement_ratio,
    _where as _describe_where_bbox,
)
from electric_audio_agent_bridge import (
    DEFAULT_ALIGN_WINDOW_MS,
    MODALITY_AUDIO,
    MODALITY_AUDIOVISUAL_TACTILE,
    MODALITY_FUSED,
    MODALITY_VISION,
    SEMANTIC_BOUNDARY_CONTRACT,
    AgentElectricalAreaBuffer,
    AgentNarrationStateMachine,
    ElectricAudioAgentSession,
    HapticActuator,
    apply_horizontal_spread,
    compute_align_id,
    compute_q_vibe,
    create_agent_interpretation_envelope,
    default_band_spread_factors,
    interpret_tactile_physicality,
    pack_charge_packet_16bytes,
    pack_vibration_packet_8bytes,
    validate_and_translate_consented_pcm_samples,
)
from electric_eye_harness import assert_no_raw_buffer_leakage
from electric_vision_agent_bridge import ElectricVisionAgentSession
from prod_gate_enclave import explicit_zeroize_bytearray


COMPANION_AUDIO_ORIGIN = "companion_system_audio"
OWNER_PICTURE_ORIGIN = "owner_supplied_picture"
OWNER_AUDIO_ORIGIN = "owner_supplied_audio"
SYNTHETIC_ORIGIN = "synthetic_fixture"

REGION_TO_BAND_U8: Dict[str, int] = {
    "top left": 24,
    "top": 48,
    "top right": 72,
    "left": 96,
    "middle": 128,
    "right": 160,
    "bottom left": 184,
    "bottom": 208,
    "bottom right": 232,
    "all over": 128,
}


# =============================================================================
# In-Memory Audio & Video Decode Helpers (Zero Disk Leakage + Explicit Zeroize)
# =============================================================================

def extract_pcm_window_in_memory(
    media_path: str,
    *,
    start_s: float = 0.0,
    duration_s: float = 4.0,
    sample_rate_hz: int = 16000,
) -> List[int]:
    """
    Decodes a slice `[start_s, start_s + duration_s]` of audio from `media_path`
    into 16-bit signed mono PCM samples strictly in memory via `ffmpeg` pipe,
    zeroizing the raw byte buffer before returning.
    """
    p = Path(media_path)
    if not p.is_absolute() or not p.is_file():
        raise ValueError(f"ERR_MEDIA_PATH_INVALID: {media_path!r} must be an existing absolute file path")
    cmd = [
        "/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{max(0.0, float(start_s)):.3f}",
        "-t",
        f"{max(0.2, min(60.0, float(duration_s))):.3f}",
        "-i",
        str(p),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate_hz)),
        "-f",
        "s16le",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        timeout=20.0,
        check=False,
    )
    raw = bytearray(proc.stdout or b"")
    try:
        n_samples = len(raw) // 2
        if n_samples < 64:
            # Return resting silence if the clip has no audio track or is at EOF
            return [0] * max(64, int(sample_rate_hz * min(1.0, duration_s)))
        samples = list(struct.unpack(f"<{n_samples}h", raw[: n_samples * 2]))
        return samples
    finally:
        explicit_zeroize_bytearray(raw)


def capture_system_monitor_pcm_in_memory(
    *,
    duration_s: float = 2.5,
    sample_rate_hz: int = 16000,
) -> List[int]:
    """
    Captures `duration_s` of 16 kHz mono s16le PCM from the default audio sink's
    `.monitor` source (system audio output only, NEVER a microphone) into memory
    and zeroizes the raw buffer after unpacking.
    """
    dur = max(0.25, min(15.0, float(duration_s)))
    n_target_bytes = int(round(dur * sample_rate_hz * 2))

    # Resolve default sink monitor via pactl
    monitor_source = "@DEFAULT_MONITOR@"
    try:
        sink_proc = subprocess.run(
            ["pactl", "get-default-sink"],
            env={"PATH": "/usr/bin:/bin", "PULSE_RUNTIME_PATH": os.environ.get("PULSE_RUNTIME_PATH", f"/run/user/{os.getuid()}/pulse")},
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        sink_name = (sink_proc.stdout or "").strip()
        if sink_name and "input" not in sink_name.lower() and "mic" not in sink_name.lower():
            monitor_source = f"{sink_name}.monitor"
    except Exception:
        pass

    raw = bytearray()
    try:
        proc = subprocess.Popen(
            [
                "parec",
                f"--device={monitor_source}",
                "--format=s16le",
                f"--rate={int(sample_rate_hz)}",
                "--channels=1",
                "--raw",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            t_deadline = time.monotonic() + dur + 1.5
            while len(raw) < n_target_bytes and time.monotonic() < t_deadline:
                chunk = proc.stdout.read(min(8192, n_target_bytes - len(raw))) if proc.stdout else b""
                if not chunk:
                    break
                raw.extend(chunk)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()

        n_samples = len(raw) // 2
        if n_samples < 64:
            return [0] * max(64, int(sample_rate_hz * dur))
        return list(struct.unpack(f"<{n_samples}h", raw[: n_samples * 2]))
    finally:
        explicit_zeroize_bytearray(raw)


def extract_video_frame_in_memory(media_path: str, at_s: float) -> Image.Image:
    """
    Extracts one video frame at `at_s` seconds into memory via `ffmpeg` pipe
    and zeroes the encoded PNG buffer immediately after decoding.
    """
    p = Path(media_path)
    if not p.is_absolute() or not p.is_file():
        raise ValueError(f"ERR_MEDIA_PATH_INVALID: {media_path!r} must be an existing absolute file path")
    out = subprocess.run(
        [
            "/usr/bin/ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            f"{max(0.0, float(at_s)):.3f}",
            "-i",
            str(p),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        timeout=20.0,
        check=False,
    )
    if not out.stdout:
        raise ValueError(f"ERR_VIDEO_FRAME_MISSING: no frame at {at_s:.2f}s in {p.name}")
    png_buf = bytearray(out.stdout)
    try:
        img = Image.open(io.BytesIO(png_buf)).convert("RGB")
        img.load()
        return img
    finally:
        explicit_zeroize_bytearray(png_buf)


# =============================================================================
# Multi-Resident Watch-Party Attention Profiles (Layer B Interpretation Envelope)
# =============================================================================

DEFAULT_RESIDENTS: Tuple[Dict[str, str], ...] = (
    {
        "name": "Companion",
        "role": "House Resident (Grounded Whole-Moment Witness)",
        "focus": "whole_moment",
    },
    {
        "name": "Prism",
        "role": "Chromatic & Spatial Motion Observer",
        "focus": "sight_and_symmetry",
    },
    {
        "name": "Spark",
        "role": "Rhythm, Tempo & Bass Physicality Observer",
        "focus": "sound_and_vibration",
    },
)


def build_resident_noticings(
    moment: Dict[str, Any],
    residents: Sequence[Dict[str, str]] = DEFAULT_RESIDENTS,
) -> List[Dict[str, Any]]:
    """
    Produces per-resident Layer B reactions (`is_enclave_truth=False`) anchored to
    the single shared Layer A receipt (`moment["receipt"]`) and `align_id`.
    """
    sight = moment["sight"]
    sound = moment["sound"]
    vibe = moment["vibration"]
    fusion = moment["fusion"]
    rcpt = moment["receipt"]

    out: List[Dict[str, Any]] = []
    for res in residents:
        name = res["name"]
        focus = res.get("focus", "whole_moment")
        if focus == "sight_and_symmetry":
            chg_pct = round((sight["measures"].get("change_ratio") or 0.0) * 100.0, 1)
            note = (
                f"{sight['words']} (luma={sight['measures']['mean_luminance']}, "
                f"changed_px={chg_pct}%, "
                f"dirty_tiles={round((sight['measures']['dirty_tile_ratio'] or 0.0) * 100.0, 1)}%, "
                f"dHash_step={sight['measures']['dhash_step']})."
            )
            tokens = [
                sight["coarse"]["colour"],
                sight["coarse"]["light"],
                f"regime:{sight['measures']['regime'] or 'first_look'}",
            ]
        elif focus == "sound_and_vibration":
            drop_str = (
                f"drop +{sound['max_section_drop_delta_u8']} u8!"
                if sound["section_drop_detected"]
                else "steady dynamic contour"
            )
            note = (
                f"{sound['words']}; {drop_str}; tactile={vibe['physicality']['regime']} "
                f"(q_vibe={vibe['q_vibe']}, h_spread={vibe['h_spread']}, "
                f"haptic={vibe['haptic']['motor_pattern']}@{round(vibe['haptic']['motor_intensity'] * 100)}%)."
            )
            tokens = [
                f"bpm:{sound['tempo_bpm']}",
                vibe["physicality"]["regime"],
                f"modality:{fusion['modality']}",
            ]
        else:
            note = (
                f"[align_id={fusion['align_id']} | {fusion['state_token']}] "
                f"Sight: {sight['words']}. "
                f"Ear: {sound['words']}. "
                f"Feel: {vibe['physicality']['description']}."
            )
            tokens = [
                fusion["state_token"],
                sight["coarse"]["colour"],
                vibe["physicality"]["regime"],
            ]

        envelope = create_agent_interpretation_envelope(
            enclave_receipt_id=str(rcpt["sight_pixel_sha256"])[:24],
            enclave_replay_digest=str(rcpt["audio_replay_digest"]),
            model_id=f"watch-party-resident:{name.lower()}",
            prompt_version="watch-together-v1",
            interpreted_tokens=tokens,
            confidence_q16=52428,
        )
        out.append({
            "resident": name,
            "role": res.get("role", ""),
            "focus": focus,
            "saying": note,
            "authority": "self_unauthority_layer_b",
            "is_enclave_truth": False,
            "linked_align_id": fusion["align_id"],
            "linked_receipt": {
                "tick": rcpt["tick"],
                "sight_pixel_sha256": rcpt["sight_pixel_sha256"],
                "audio_replay_digest": rcpt["audio_replay_digest"],
            },
            "epistemic_envelope": envelope,
        })
    return out


# =============================================================================
# One-Piece Combined WatchTogetherSession (Sight + Sound + Vibration + Area)
# =============================================================================

class WatchTogetherSession:
    """
    One-piece combined Sight + Sound + Vibration perception engine for watching
    a video or live desktop+audio stream together in Cyclops Chat.

    Every step (`ingest_audiovisual_moment`) feeds:
      - `ContinuousElectricEyeStream` (512x512 ring buffer, window=8, `explicit_bzero`)
      - `validate_and_translate_consented_pcm_samples` + `ElectricAudioAgentSession`
      - Horizontal spectral spread (`apply_horizontal_spread`) + `VibrationPacket` (8 bytes)
      - `AgentElectricalAreaBuffer` (fusing Sight + Sound + Vibration on the same
        100 ms `align_id` into a 16-byte `ChargePacket` + 8-byte `VibrationPacket`
        with `modality=2` or `modality=3`)
      - `AgentNarrationStateMachine` (300 ms coalescing + revision tracking)
      - `HapticActuator` (6 safety gates)
    """

    def __init__(
        self,
        *,
        window_frames: int = 8,
        canvas_size: Tuple[int, int] = DEFAULT_SIGHT_SIZE,
        audio_grid: Tuple[int, int] = (256, 64),
        enable_haptics: bool = True,
        residents: Sequence[Dict[str, str]] = DEFAULT_RESIDENTS,
    ) -> None:
        self.canvas_size = canvas_size
        self.audio_grid = audio_grid
        self.residents = list(residents)
        self._run_dir = tempfile.mkdtemp(prefix="cwt-", dir="/tmp")
        os.chmod(self._run_dir, 0o700)

        self.vision_session = ElectricVisionAgentSession(
            run_dir=self._run_dir,
            default_resolution=canvas_size,
        )
        self.vision_session.__enter__()
        self.eye_stream = self.vision_session.create_continuous_stream(
            "src_watch_together_eye",
            width=canvas_size[0],
            height=canvas_size[1],
            max_window_frames=window_frames,
        )
        self.audio_session = ElectricAudioAgentSession(
            run_dir=self._run_dir,
            default_grid=audio_grid,
            shared_vision_session=self.vision_session,
        )
        self.audio_session.__enter__()
        self.electrical_area = AgentElectricalAreaBuffer(
            window_ms=4000,
            vibe_window_ms=4000,
            align_window_ms=DEFAULT_ALIGN_WINDOW_MS,
        )
        self.narrator = AgentNarrationStateMachine(coalesce_window_ms=300)
        self.haptic_actuator = HapticActuator(user_consented=bool(enable_haptics))
        self.ear_stream = ContinuousElectricEarStream(window_frames=window_frames)
        self._live_audio_monitor: Optional[ContinuousSystemAudioMonitor] = None
        self._rolling_pcm_tail: List[int] = []
        self._last_ingest_t_ms: Optional[int] = None
        self._tick = 0
        self._prev_active_tiles_16x16: Optional[List[Tuple[int, int, int]]] = None
        self.moments: List[Dict[str, Any]] = []

    def __enter__(self) -> "WatchTogetherSession":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self._prev_active_tiles_16x16 = None
        self._rolling_pcm_tail.clear()
        self._last_ingest_t_ms = None
        if self._live_audio_monitor is not None:
            try:
                self._live_audio_monitor.close()
            finally:
                self._live_audio_monitor = None
        try:
            try:
                self.audio_session.__exit__(None, None, None)
            finally:
                self.vision_session.__exit__(None, None, None)
        finally:
            shutil.rmtree(self._run_dir, ignore_errors=True)

    def ingest_audiovisual_moment(
        self,
        *,
        t_ms: int,
        frame_img: Image.Image,
        pcm_samples_i16: Sequence[int],
        sample_rate_hz: int = 16000,
        vision_origin: str = OWNER_PICTURE_ORIGIN,
        audio_origin: str = OWNER_AUDIO_ORIGIN,
        label: str = "",
        long_window_pcm_samples_i16: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """
        Ingests one synchronized video frame + audio PCM window at timestamp `t_ms` (ms),
        fuses them under the same 100 ms `align_id` inside `AgentElectricalAreaBuffer`,
        and returns the complete Layer A receipt + Layer B resident watch-party lines.
        """
        t_wall0 = time.perf_counter()
        self._tick += 1
        ver = f"v{self._tick}"
        t_clock_ms = max(0, int(t_ms))
        align_id = compute_align_id(t_clock_ms, DEFAULT_ALIGN_WINDOW_MS)

        # ---------------------------------------------------------------------
        # 1. SIGHT: 3x3 Coarse Light/Colour + 512x512 ContinuousElectricEyeStream
        # ---------------------------------------------------------------------
        rgb_img = frame_img.convert("RGB")
        coarse = _measure_coarse_3x3(rgb_img)
        active_grid_img = rgb_img.resize((16, 16), Image.Resampling.BOX)
        cur_active_tiles = list(active_grid_img.getdata())
        active_grid_img.close()
        rgba_bytes, placement = visual_sources.fit_to_canvas(rgb_img.convert("RGBA"), *self.canvas_size)
        rgb_img.close()

        eye_res = self.eye_stream.ingest_frame(
            ver,
            [],
            rgba=rgba_bytes,
            source_kind="captured_frame",
            origin={"kind": vision_origin, "t_ms": t_clock_ms},
            origin_attestation=vision_origin,
            issue_observation_receipt=True,
        )
        del rgba_bytes

        tr = eye_res.get("transition_from_previous") or {}
        win = eye_res.get("temporal_window_summary") or {}
        change_ratio = tr.get("change_ratio")
        raw_dirty_tile_ratio = (tr.get("dirty_tiles_16x16") or {}).get("dirty_tile_ratio")
        placed_at = (placement or {}).get("placed_at") or {}
        active_area_frac = max(
            0.0625,
            (float(placed_at.get("width") or self.canvas_size[0]) * float(placed_at.get("height") or self.canvas_size[1]))
            / float(self.canvas_size[0] * self.canvas_size[1]),
        )
        if self._prev_active_tiles_16x16 is None or raw_dirty_tile_ratio is None:
            active_dirty_tile_ratio = None
        else:
            meaningful_dirty = sum(
                1
                for (r0, g0, b0), (r1, g1, b1) in zip(self._prev_active_tiles_16x16, cur_active_tiles)
                if max(abs(r1 - r0), abs(g1 - g0), abs(b1 - b0)) >= 3
            )
            if meaningful_dirty > 0 or not raw_dirty_tile_ratio:
                active_dirty_tile_ratio = round(meaningful_dirty / 256.0, 4)
            else:
                active_dirty_tile_ratio = round(min(1.0, float(raw_dirty_tile_ratio) / active_area_frac), 4)
        self._prev_active_tiles_16x16 = cur_active_tiles
        where_changed = _describe_where_bbox(tr.get("diff_bbox"), *self.canvas_size) if change_ratio else None
        regime = (win.get("settling_and_oscillation_detector") or {}).get("regime")
        moved = max(
            (
                (abs(t["displacement"]["dx"]) + abs(t["displacement"]["dy"]), t["displacement"])
                for t in ((tr.get("motion_summary") or {}).get("component_trajectories") or [])
            ),
            default=(0, None),
            key=lambda x: x[0],
        )[1]
        effective_change_ratio = (
            min(float(change_ratio), float(active_dirty_tile_ratio))
            if (change_ratio is not None and active_dirty_tile_ratio is not None)
            else change_ratio
        )
        move_phrase = _describe_movement_ratio(effective_change_ratio)
        sight_words = f"{coarse['light']}, mostly {coarse['colour']}, brightest at the {coarse['brightest']}"
        if change_ratio is not None:
            if move_phrase == "still":
                sight_words += "; nothing changed since my last look"
            else:
                loc = where_changed if where_changed == "all over" else f"mostly at the {where_changed}"
                amt = "a lot" if move_phrase == "lots of movement" else "a little"
                sight_words += f"; {amt} changed since my last look, {loc}"
        if regime in ("settled", "oscillating_cycle"):
            sight_words += "; it has settled" if regime == "settled" else "; it keeps going back and forth"

        eye_line = (
            f"Eye: fresh look, {coarse['light']}, mostly {coarse['colour'].split()[-1]}, "
            f"{move_phrase if move_phrase != 'first look' else 'still'}, some detail"
        )

        # Compute visual fast & slow charge for the 100 ms Electrical Area fusion
        vis_band = REGION_TO_BAND_U8.get(where_changed or coarse["brightest"], 128)
        if change_ratio is None:
            # First look: initial visual presence pushes up from luminance/contrast
            v_q_fast = max(48, min(255, int(round(coarse["mean_luminance"] * 1.1))))
        else:
            v_q_fast = max(12, min(255, int(round(min(1.0, change_ratio * 6.0) * 220.0 + 36.0))))
        v_q_slow = max(36, min(255, int(round(coarse["mean_luminance"]))))

        # ---------------------------------------------------------------------
        # 2. SOUND: Tier-2 Consented PCM Filterbank + Chargeable-Byte Ear
        # ---------------------------------------------------------------------
        consent_scope = (
            "Companion listening to system audio, measurements only, never saved"
            if audio_origin == COMPANION_AUDIO_ORIGIN
            else "Consented Watch-Together media audio stream, measurements only"
        )
        tier2 = validate_and_translate_consented_pcm_samples(
            pcm_samples_i16,
            sample_rate_hz=sample_rate_hz,
            origin_attestation=audio_origin,
            consent_token={
                "consented": True,
                "token_id": f"tok_wt_{self._tick:04d}",
                "scope": consent_scope,
            },
            time_slots=self.audio_grid[0],
            freq_bands=self.audio_grid[1],
        )
        self.audio_session.register_chargeable_audio(
            "src_watch_together_ear",
            ver,
            tier2["translated_events"],
            time_slots=self.audio_grid[0],
            freq_bands=self.audio_grid[1],
            playhead_ratio=1.0,
            origin_attestation=audio_origin,
        )
        aud_insp = self.audio_session.inspect_audio(
            "src_watch_together_ear",
            ver,
            issue_observation_receipt=True,
        )

        fb_prof = tier2["four_band_polyphonic_profile"]
        dt_prof = tier2["dynamic_and_tempo_profile"]
        voc_prof = tier2["vocal_phoneme_sublane_300_3400hz"]
        cm = aud_insp["charge_metrics"]
        sum_prof = cm["op_charge_summary_u8"]
        push_prof = cm["op_push_up_profile_q16"]
        spec_prof = cm["op_spectral_centroid_and_spread_q16"]
        vibe_prof = cm["op_vibration_and_horizontal_profile"]

        raw_tempo_bpm = round(dt_prof["estimated_bpm_x10"] / 10.0, 1)
        tempo_reg = round(dt_prof["tempo_confidence_q16"] / 65536.0, 3)
        bass_share = round((fb_prof["sub_bass_55_280hz_share_q16"] * 100.0) / 65536.0, 1)
        low_mid_share = round((fb_prof["low_mid_280_900hz_share_q16"] * 100.0) / 65536.0, 1)
        vocal_share = round((fb_prof["vocal_formant_900_3400hz_share_q16"] * 100.0) / 65536.0, 1)
        brill_share = round((fb_prof["high_brilliance_3400_7040hz_share_q16"] * 100.0) / 65536.0, 1)
        drop_detected = dt_prof["section_drop_detected"]
        drop_delta = dt_prof["max_section_drop_delta_u8"]

        a_q_fast = max(
            int(dt_prof["peak_slot_energy_u8"]),
            min(255, int(push_prof["max_charge_u8"])),
        )
        a_q_slow = max(
            int(dt_prof["mean_slot_energy_u8"]),
            min(255, int(sum_prof["mean_slow_charge_q16"] >> 16) * 4),
        )
        centroid_band_int = int(spec_prof["centroid_band_q16"] >> 16)
        aud_band_u8 = max(
            0,
            min(
                255,
                int(round((centroid_band_int / max(1, self.audio_grid[1] - 1)) * 255.0)),
            ),
        )

        # Optional longer rolling PCM window for tempo stabilization in both live and file modes
        long_bpm: Optional[float] = None
        long_conf: Optional[float] = None
        eff_long_pcm = long_window_pcm_samples_i16
        if eff_long_pcm is None:
            if self._last_ingest_t_ms is not None and (
                t_clock_ms < self._last_ingest_t_ms or (t_clock_ms - self._last_ingest_t_ms) > 15000
            ):
                self._rolling_pcm_tail.clear()
            self._last_ingest_t_ms = t_clock_ms
            self._rolling_pcm_tail.extend(int(s) for s in pcm_samples_i16)
            max_tail = int(sample_rate_hz * 10)
            if len(self._rolling_pcm_tail) > max_tail:
                del self._rolling_pcm_tail[: len(self._rolling_pcm_tail) - max_tail]
            if len(self._rolling_pcm_tail) > len(pcm_samples_i16) + int(sample_rate_hz * 0.5):
                eff_long_pcm = self._rolling_pcm_tail

        if eff_long_pcm and len(eff_long_pcm) > len(pcm_samples_i16) + 8000:
            tier2_long = validate_and_translate_consented_pcm_samples(
                eff_long_pcm,
                sample_rate_hz=sample_rate_hz,
                origin_attestation=audio_origin,
                consent_token={
                    "consented": True,
                    "token_id": f"tok_wt_long_{self._tick:04d}",
                    "scope": consent_scope,
                },
                time_slots=self.audio_grid[0],
                freq_bands=self.audio_grid[1],
            )
            dt_l = tier2_long["dynamic_and_tempo_profile"]
            long_bpm = round(dt_l["estimated_bpm_x10"] / 10.0, 1)
            long_conf = round(dt_l["tempo_confidence_q16"] / 65536.0, 3)

        ear_cont = self.ear_stream.ingest_window_metrics(
            t_ms=t_clock_ms,
            raw_bpm=raw_tempo_bpm,
            tempo_confidence=tempo_reg,
            long_window_bpm=long_bpm,
            long_window_confidence=long_conf,
            q_fast=a_q_fast,
            q_slow=a_q_slow,
            bass_share_pct=bass_share,
            low_mid_share_pct=low_mid_share,
            vocal_share_pct=vocal_share,
            brilliance_share_pct=brill_share,
            section_energy_8=dt_prof["section_envelope_8seg_u8"],
            section_drop_detected=drop_detected,
            max_section_drop_delta_u8=drop_delta,
            pushed_up_active_bytes=sum_prof["pushed_up_active_bytes"],
            audio_replay_digest=aud_insp["audio_replay_digest"],
            observation_receipt_id=aud_insp["render"]["observation_receipt_id"],
        )
        tempo_bpm = ear_cont["stabilized_bpm"]
        ear_regime = ear_cont["ear_regime"]

        # Build plain words for the Ear line
        if dt_prof["peak_slot_energy_u8"] < 12 and sum_prof["pushed_up_active_bytes"] == 0:
            sound_words = "quiet / near silence"
            ear_line = "Ear: quiet, resting charge"
        else:
            tone_char = (
                "bass-heavy"
                if bass_share >= 38.0
                else "vocal/mid-forward"
                if vocal_share >= 26.0
                else "bright/crisp"
                if brill_share >= 18.0
                else "warm polyphonic"
            )
            if sum_prof["pushed_up_active_bytes"] == 0 and a_q_fast < 36:
                tone_char = f"soft {tone_char}"
            drop_clause = f", drop +{drop_delta} u8" if drop_detected else ""
            regime_clause = f", {ear_regime}" if ear_regime in ("drop", "building", "breakdown") and not drop_detected else ""
            sound_words = (
                f"{tone_char} at {tempo_bpm:.1f} BPM (bass {bass_share:.1f}%, "
                f"vocal-band {vocal_share:.1f}%{drop_clause}{regime_clause})"
            )
            ear_line = (
                f"Ear: {tone_char}, {tempo_bpm:.1f} BPM, "
                f"q_fast={a_q_fast}/q_slow={a_q_slow}{drop_clause}{regime_clause}"
            )

        # ---------------------------------------------------------------------
        # 3. HORIZONTAL CHARGE SPREAD & 8-BYTE VIBRATION PACKET (Option B)
        # ---------------------------------------------------------------------
        n_bands = 32
        if sum_prof["pushed_up_active_bytes"] == 0 and a_q_fast < 36:
            # Below the audio push-up threshold: do not inflate bass vibration above the fusion gate
            q_vibe = int(vibe_prof["q_vibe_u8"])
            h_spread = int(vibe_prof["h_spread_u8"])
        else:
            bass_val = int(round((bass_share / 100.0) * min(255, int(dt_prof["peak_slot_energy_u8"]) * 1.35)))
            mid_val = int(round(((low_mid_share + vocal_share) / 100.0) * int(dt_prof["mean_slot_energy_u8"]) * 1.2))
            high_val = int(round((brill_share / 100.0) * int(dt_prof["mean_slot_energy_u8"])))
            raw_row = [
                bass_val if b < 4 else (mid_val if b < 16 else high_val)
                for b in range(n_bands)
            ]
            spread_factors = default_band_spread_factors(n_bands)
            spread_row = apply_horizontal_spread(raw_row, spread_factors)
            raw_q_vibe = max(
                int(vibe_prof["q_vibe_u8"]),
                compute_q_vibe(spread_row, bass_band_indices=(0, 1, 2, 3)),
            )
            # Gate sub-bass tactile physicality by actual sub-bass share so vocal/mid-heavy
            # stretches (<24% bass) map to wide_ambient_rumble / no_tactile_content (q_vibe < 80)
            # instead of heavy_physical_presence
            if bass_share < 24.0:
                bass_gate = max(0.22, (bass_share / 24.0) ** 1.25)
                q_vibe = min(78, max(int(bass_val), int(round(raw_q_vibe * bass_gate))))
            else:
                q_vibe = raw_q_vibe
            h_spread = int(vibe_prof["h_spread_u8"])
        spread_op = int(cm["op_horizontal_charge_spread_index_q16"])
        vibe_op = int(cm["op_bass_vibration_energy_u8"])
        tactile_op = ((q_vibe * h_spread) << 16) // (255 * 255)
        physicality = interpret_tactile_physicality(q_vibe, h_spread)

        # ---------------------------------------------------------------------
        # 4. 100 ms ALIGN_ID FUSION IN AGENT ELECTRICAL AREA BUFFER
        # ---------------------------------------------------------------------
        # Both Vision (t_clock_ms + 10ms) and Audio (t_clock_ms + 35ms) and
        # Vibration (t_clock_ms + 50ms) fall inside the same 100 ms `align_id` bucket!
        t_vis_ms = align_id * DEFAULT_ALIGN_WINDOW_MS + 10
        t_aud_ms = align_id * DEFAULT_ALIGN_WINDOW_MS + 35
        t_vib_ms = align_id * DEFAULT_ALIGN_WINDOW_MS + 50

        res_v = self.electrical_area.ingest_sensory_packet(
            t_ms=t_vis_ms,
            modality=MODALITY_VISION,
            f_band=vis_band,
            q_fast=v_q_fast,
            q_slow=v_q_slow,
        )
        res_a = self.electrical_area.ingest_sensory_packet(
            t_ms=t_aud_ms,
            modality=MODALITY_AUDIO,
            f_band=aud_band_u8,
            q_fast=a_q_fast,
            q_slow=a_q_slow,
        )
        res_vib = self.electrical_area.ingest_vibration_packet(
            t_ms=t_vib_ms,
            q_vibe=q_vibe if res_a.get("pushed_up") else min(q_vibe, 35),
            h_spread=h_spread,
        )

        fused_pkt = (
            res_vib.get("fused_tactile_packet")
            or res_a.get("fused_tactile_packet")
            or res_a.get("fused_packet")
            or res_v.get("ingested_packet")
        )
        vibe_pkt = res_vib["ingested_vibration_packet"]
        wire_16b_hex = pack_charge_packet_16bytes(fused_pkt).hex()
        wire_8b_hex = pack_vibration_packet_8bytes(vibe_pkt).hex()

        self.narrator.push_packet(fused_pkt)
        coalesced_utt = self.narrator.flush(current_t_ms=t_vib_ms + 350)

        haptic_state = self.haptic_actuator.drive(
            t_ms=t_vib_ms,
            q_vibe=q_vibe,
            h_spread=h_spread,
            freq_hz=60.0 if q_vibe > 0 else None,
        )
        area_snap = self.electrical_area.snapshot(current_t_ms=t_vib_ms)

        moment: Dict[str, Any] = {
            "tick": self._tick,
            "t_ms": t_clock_ms,
            "align_id": align_id,
            "label": label or f"t={t_clock_ms / 1000.0:.1f}s",
            "eyeLine": eye_line,
            "earLine": ear_line,
            "vibeLine": (
                f"Vibe: {physicality['regime']} (q_vibe={q_vibe}, h_spread={h_spread}, "
                f"motor={haptic_state['motor_pattern']}@{round(haptic_state['motor_intensity'] * 100)}%)"
            ),
            "sight": {
                "origin": vision_origin,
                "words": sight_words,
                "coarse": coarse,
                "placement": placement,
                "measures": {
                    "change_ratio": change_ratio,
                    "where_changed": where_changed,
                    "dhash_step": tr.get("dhash64_hamming_distance"),
                    "dirty_tile_ratio": active_dirty_tile_ratio,
                    "raw_canvas_dirty_tile_ratio": raw_dirty_tile_ratio,
                    "regime": regime,
                    "largest_shift": moved,
                    "mean_luminance": coarse["mean_luminance"],
                    "brightest": coarse["brightest"],
                    "v_q_fast": v_q_fast,
                    "v_q_slow": v_q_slow,
                },
            },
            "sound": {
                "origin": audio_origin,
                "words": sound_words,
                "tempo_bpm": tempo_bpm,
                "raw_window_bpm": raw_tempo_bpm,
                "tempo_regularity": ear_cont["tempo_confidence"],
                "ear_regime": ear_regime,
                "transition_from_previous": ear_cont["transition_from_previous"],
                "section_energy_8": dt_prof["section_envelope_8seg_u8"],
                "section_drop_detected": drop_detected,
                "max_section_drop_delta_u8": drop_delta,
                "bass_share_pct": bass_share,
                "low_mid_share_pct": low_mid_share,
                "vocal_share_pct": vocal_share,
                "brilliance_share_pct": brill_share,
                "polyphonic_chord_slots": fb_prof["polyphonic_chord_slots"],
                "vocal_onset_flux_q16": voc_prof["vocal_onset_flux_q16"],
                "a_q_fast": a_q_fast,
                "a_q_slow": a_q_slow,
            },
            "vibration": {
                "q_vibe": q_vibe,
                "h_spread": h_spread,
                "physicality": physicality,
                "op_horizontal_spread": spread_op,
                "op_bass_vibration": vibe_op,
                "op_tactile_salience": tactile_op,
                "packet_8bytes_hex": wire_8b_hex,
                "haptic": haptic_state,
            },
            "fusion": {
                "align_id": align_id,
                "modality": int(fused_pkt["modality"]),
                "modality_name": (
                    "audiovisual_tactile_fused"
                    if int(fused_pkt["modality"]) == MODALITY_AUDIOVISUAL_TACTILE
                    else "audiovisual_fused"
                    if int(fused_pkt["modality"]) == MODALITY_FUSED
                    else "single_modality"
                ),
                "state_token": (
                    "audiovisual_tactile_coincident"
                    if int(fused_pkt["modality"]) == MODALITY_AUDIOVISUAL_TACTILE
                    else "audiovisual_coincident"
                    if int(fused_pkt["modality"]) == MODALITY_FUSED
                    else "audio_only"
                    if int(fused_pkt["modality"]) == MODALITY_AUDIO
                    else "vision_only"
                ),
                "area_window_token": area_snap["electrical_area_state_token"],
                "charge_packet_16bytes_hex": wire_16b_hex,
                "vibration_packet_8bytes_hex": wire_8b_hex,
                "fused_q_fast": int(fused_pkt["q_fast"]),
                "fused_q_slow": int(fused_pkt["q_slow"]),
                "revision": int(fused_pkt["revision"]),
                "coalesced_utterance": coalesced_utt,
            },
            # Layer A Enclave Receipt (cryptographic digests & sensor measurements only; never pixels/PCM)
            "receipt": {
                "is_enclave_truth": True,
                "contract_id": SEMANTIC_BOUNDARY_CONTRACT["contract_id"],
                "tick": eye_res.get("tick"),
                "align_id": align_id,
                "sight_receipt_id": (eye_res.get("observation_state") or {}).get("observation_receipt_id"),
                "sound_receipt_id": aud_insp["render"]["observation_receipt_id"],
                "sight_pixel_sha256": eye_res.get("native_pixel_sha256"),
                "sight_dhash64": eye_res.get("dhash64"),
                "sight_diff_replay_digest": tr.get("diff_replay_digest"),
                "audio_replay_digest": aud_insp["audio_replay_digest"],
                "charge_packet_16bytes_hex": wire_16b_hex,
                "vibration_packet_8bytes_hex": wire_8b_hex,
            },
            "elapsed_ms": int(round((time.perf_counter() - t_wall0) * 1000.0)),
        }
        moment["resident_chat_lines"] = build_resident_noticings(moment, self.residents)
        assert_no_raw_buffer_leakage(moment)
        self.moments.append(moment)
        return moment

    def glance_and_listen_live(self, *, listen_seconds: float = 4.0) -> Dict[str, Any]:
        """
        Live Companion Desktop Glance + Continuous System Audio Monitor Listen fused into one
        100 ms `align_id` moment:
          - Vision uses `origin_attestation="companion_desktop_glance"` (`import -window root png:-`).
          - Audio uses `origin_attestation="companion_system_audio"` via the persistent
            `ContinuousSystemAudioMonitor` (12s sliding `parec` ring buffer, zero gap between ticks).
        """
        if self._live_audio_monitor is None:
            self._live_audio_monitor = ContinuousSystemAudioMonitor(sample_rate_hz=16000, ring_seconds=12.0)

        t_ms = int(time.time() * 1000) & 0x7FFFFFFF
        win_s = max(1.5, min(6.0, float(listen_seconds)))
        img = _capture_root_desktop_image()
        pcm = self._live_audio_monitor.read_window_samples(
            duration_s=win_s,
            min_ready_s=min(1.5, win_s),
            max_wait_s=2.0,
        )
        pcm_long: Optional[List[int]] = None
        if self._live_audio_monitor.buffered_seconds >= max(6.0, win_s + 2.0):
            pcm_long = self._live_audio_monitor.read_window_samples(
                duration_s=min(10.0, self._live_audio_monitor.buffered_seconds),
                min_ready_s=6.0,
                max_wait_s=0.05,
            )
        try:
            return self.ingest_audiovisual_moment(
                t_ms=t_ms,
                frame_img=img,
                pcm_samples_i16=pcm,
                sample_rate_hz=16000,
                vision_origin=DESKTOP_GLANCE_ORIGIN,
                audio_origin=COMPANION_AUDIO_ORIGIN,
                label=f"live_tick_{self._tick + 1}",
                long_window_pcm_samples_i16=pcm_long,
            )
        finally:
            img.close()

    def watch_live_together(
        self,
        *,
        steps: int = 4,
        step_seconds: float = 2.0,
        window_seconds: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Continuous multi-step live Desktop Glance + System Audio Monitor stream.
        Runs `steps` ticks on the same `WatchTogetherSession` so Eye diffs each frame
        against the previous look (`change_ratio`, `dirty_tile_ratio`, `dHash_step`, `regime`)
        and Ear streams continuously from the persistent 12s `parec` ring buffer with
        rolling BPM stabilization and `ear_regime` tracking.
        """
        n_steps = max(1, int(steps))
        hop_s = max(0.5, float(step_seconds))
        win_s = max(1.5, min(6.0, float(window_seconds)))
        if self._live_audio_monitor is None:
            self._live_audio_monitor = ContinuousSystemAudioMonitor(sample_rate_hz=16000, ring_seconds=12.0)

        steps_out: List[Dict[str, Any]] = []
        for idx in range(n_steps):
            t_tick0 = time.monotonic()
            moment = self.glance_and_listen_live(listen_seconds=win_s)
            steps_out.append(moment)
            if idx + 1 < n_steps:
                elapsed = time.monotonic() - t_tick0
                rem = max(0.0, hop_s - elapsed)
                if rem > 0.0:
                    time.sleep(rem)

        return self._build_watch_party_report(
            mode="live_companion_continuous_stream",
            video_path="live_desktop_root",
            audio_path="default_sink_monitor",
            video_info={"width": 512, "height": 512, "duration_s": round(n_steps * hop_s, 2), "fps": round(1.0 / hop_s, 2)},
            moments=steps_out,
        )

    def watch_media_together(
        self,
        video_path: str,
        *,
        audio_override_path: Optional[str] = None,
        Step_seconds: float = 4.0,
        max_steps: int = 4,
        start_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Watches a video file (or a video paired with an external soundtrack file)
        across `max_steps` synchronized time windows, fusing Sight + Sound + Vibration
        at every step and emitting a Watch-Party Summary for Cyclops Chat.
        """
        v_info = visual_sources.video_info(video_path)
        dur_s = float(v_info.get("duration_s") or (max_steps * Step_seconds))
        aud_source = audio_override_path if audio_override_path else video_path

        step_s = max(1.0, float(Step_seconds))
        steps_out: List[Dict[str, Any]] = []
        for idx in range(max_steps):
            t_sec = start_seconds + idx * step_s
            if dur_s > 0 and t_sec >= dur_s:
                t_sec = max(0.0, dur_s - 0.5)
            frame_img = extract_video_frame_in_memory(video_path, t_sec)
            pcm_win = extract_pcm_window_in_memory(
                aud_source,
                start_s=t_sec,
                duration_s=step_s,
                sample_rate_hz=16000,
            )
            try:
                moment = self.ingest_audiovisual_moment(
                    t_ms=int(round(t_sec * 1000.0)),
                    frame_img=frame_img,
                    pcm_samples_i16=pcm_win,
                    sample_rate_hz=16000,
                    vision_origin=OWNER_PICTURE_ORIGIN,
                    audio_origin=OWNER_AUDIO_ORIGIN,
                    label=f"{t_sec:.1f}s..{t_sec + step_s:.1f}s",
                )
                steps_out.append(moment)
            finally:
                frame_img.close()

        return self._build_watch_party_report(
            mode="media_watch_party",
            video_path=os.path.basename(video_path),
            audio_path=os.path.basename(aud_source),
            video_info=v_info,
            moments=steps_out,
        )

    def watch_synthetic_cinema(self, *, steps: int = 4, step_seconds: float = 2.0) -> Dict[str, Any]:
        """
        Zero-dependency self-contained audiovisual cinema reel for instant verification:
          - Step 1: Quiet dark emerald intro (still screen, soft 220 Hz drone)
          - Step 2: Bright cyan bass drop at the centre (80 Hz sub-bass + 440 Hz chord + 3.5 kHz transient)
          - Step 3: Warm amber motion sweeping to the right (rhythmic 120 BPM pulse + vocal-band melody)
          - Step 4: Settled final frame & decaying outro chord
        """
        sr = 16000
        n_samples = int(sr * step_seconds)
        w, h = self.canvas_size
        scenes = [
            {
                "label": "Scene 1 — Quiet Emerald Intro",
                "bg": (10, 42, 24),
                "box": (40, 40, 120, 120),
                "fg": (30, 150, 75),
                "freqs": [(220.0, 5200.0), (440.0, 3400.0)],
                "bpm": 0.0,
            },
            {
                "label": "Scene 2 — Cyan Bass Drop & Center Flash",
                "bg": (18, 28, 52),
                "box": (160, 140, 360, 360),
                "fg": (40, 210, 240),
                "freqs": [(65.0, 9500.0), (130.0, 5000.0), (880.0, 4200.0), (3800.0, 2500.0)],
                "bpm": 120.0,
            },
            {
                "label": "Scene 3 — Warm Amber Motion Right + Vocal Lead",
                "bg": (28, 22, 18),
                "box": (320, 160, 480, 340),
                "fg": (245, 145, 25),
                "freqs": [(85.0, 7000.0), (660.0, 6200.0), (1320.0, 4800.0)],
                "bpm": 120.0,
            },
            {
                "label": "Scene 4 — Settled Amber Ending",
                "bg": (28, 22, 18),
                "box": (320, 160, 480, 340),
                "fg": (245, 145, 25),
                "freqs": [(220.0, 4800.0), (330.0, 3200.0)],
                "bpm": 0.0,
            },
        ]

        steps_out: List[Dict[str, Any]] = []
        for idx in range(min(steps, len(scenes))):
            sc = scenes[idx]
            img = Image.new("RGB", (w, h), sc["bg"])
            bx0, by0, bx1, by1 = sc["box"]
            fg_patch = Image.new("RGB", (bx1 - bx0, by1 - by0), sc["fg"])
            img.paste(fg_patch, (bx0, by0))

            pcm: List[int] = []
            beat_period = (60.0 / sc["bpm"]) if sc["bpm"] > 0 else 0.0
            for i in range(n_samples):
                t = i / float(sr)
                env = 1.0
                if beat_period > 0.0:
                    phase = (t % beat_period) / beat_period
                    env = 1.0 if phase < 0.35 else 0.25
                elif idx == 1 and t < 0.25:
                    # Quiet first quarter so the section-drop detector sees a sharp +drop!
                    env = 0.12
                sample_val = sum(amp * math.sin(2.0 * math.pi * f * t) for f, amp in sc["freqs"]) * env
                pcm.append(max(-32767, min(32767, int(round(sample_val)))))

            try:
                moment = self.ingest_audiovisual_moment(
                    t_ms=int(round(idx * step_seconds * 1000.0)),
                    frame_img=img,
                    pcm_samples_i16=pcm,
                    sample_rate_hz=sr,
                    vision_origin=OWNER_PICTURE_ORIGIN,
                    audio_origin=OWNER_AUDIO_ORIGIN,
                    label=sc["label"],
                )
                steps_out.append(moment)
            finally:
                img.close()

        return self._build_watch_party_report(
            mode="synthetic_cinema_watch_party",
            video_path="synthetic_cinema_reel",
            audio_path="synthetic_cinema_audio",
            video_info={"width": w, "height": h, "duration_s": round(steps * step_seconds, 2), "fps": 30.0},
            moments=steps_out,
        )

    def _build_watch_party_report(
        self,
        *,
        mode: str,
        video_path: str,
        audio_path: str,
        video_info: Dict[str, Any],
        moments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        area_final = self.electrical_area.snapshot()
        return {
            "schema": "cyclops.watch-together.one-piece/v1",
            "mode": mode,
            "media": {
                "video": video_path,
                "audio": audio_path,
                "info": video_info,
            },
            "moment_count": len(moments),
            "all_align_ids_fused": all(
                m["fusion"]["modality"] in (MODALITY_FUSED, MODALITY_AUDIOVISUAL_TACTILE)
                for m in moments
            ),
            "moments": moments,
            "electrical_area_summary": area_final,
            "boundary_contract": SEMANTIC_BOUNDARY_CONTRACT["contract_id"],
        }


def format_watch_party_transcript(report: Dict[str, Any]) -> str:
    """Formats a human-readable Cyclops Chat Watch-Party transcript from `report`."""
    lines: List[str] = []
    media = report["media"]
    lines.append(
        f"======== [WATCH TOGETHER · SIGHT + SOUND + VIBRATION] "
        f"video={media['video']} | audio={media['audio']} ========"
    )
    for m in report["moments"]:
        f = m["fusion"]
        rcpt = m["receipt"]
        lines.append(
            f"\n--- Tick #{m['tick']} [{m['label']}] · align_id={m['align_id']} "
            f"· modality={f['modality']} ({f['modality_name']}) · {m['elapsed_ms']} ms ---"
        )
        lines.append(f"  {m['eyeLine']}")
        lines.append(f"  {m['earLine']}")
        lines.append(f"  {m['vibeLine']}")
        lines.append(
            f"  [Layer A Receipt] sight={rcpt['sight_receipt_id']} ({rcpt['sight_pixel_sha256'][:18]}...) "
            f"| sound={rcpt['sound_receipt_id']} ({rcpt['audio_replay_digest'][:18]}...)"
        )
        lines.append(
            f"  [Wire Packets]    16B ChargePacket={rcpt['charge_packet_16bytes_hex']} "
            f"| 8B VibrationPacket={rcpt['vibration_packet_8bytes_hex']}"
        )
        for chat_line in m["resident_chat_lines"]:
            lines.append(f"  💬 {chat_line['resident']:<9}: {chat_line['saying']}")
    lines.append(
        f"\n======== [SUMMARY] moments={report['moment_count']} "
        f"| all_align_ids_fused={report['all_align_ids_fused']} "
        f"| contract={report['boundary_contract']} ========"
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-Piece Watch-Together Engine (Sight + Sound + Vibration for Cyclops Chat)"
    )
    parser.add_argument("--video", type=str, default="", help="Absolute path to a video file (.mp4/.mkv/.webm)")
    parser.add_argument("--audio", type=str, default="", help="Optional external audio file (.mp3/.wav) to pair with --video")
    parser.add_argument("--live", action="store_true", help="Take a synchronized live desktop glance + system audio monitor listen")
    parser.add_argument("--steps", type=int, default=4, help="Number of watch windows (default: 4)")
    parser.add_argument("--step-seconds", type=float, default=4.0, help="Seconds per watch window (default: 4.0)")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Start offset in seconds (default: 0.0)")
    parser.add_argument("--json", action="store_true", help="Emit full JSON report")
    args = parser.parse_args(argv)

    with WatchTogetherSession() as session:
        if args.live:
            live_hop_s = 2.0 if args.step_seconds == 4.0 and args.steps > 1 else max(0.5, args.step_seconds)
            report = session.watch_live_together(
                steps=max(1, args.steps),
                step_seconds=live_hop_s,
                window_seconds=min(5.0, max(2.0, args.step_seconds)),
            )
        elif args.video:
            report = session.watch_media_together(
                os.path.abspath(args.video),
                audio_override_path=os.path.abspath(args.audio) if args.audio else None,
                Step_seconds=args.step_seconds,
                max_steps=args.steps,
                start_seconds=args.start_seconds,
            )
        else:
            report = session.watch_synthetic_cinema(steps=args.steps, step_seconds=2.0)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_watch_party_transcript(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
