"""Continuous system-audio hearing for the Companion & Watch-Together: Electric Ear on live sink monitor.

On the owner's word (2026-11-04: "yea sound all good" / "Each live Ear listen hears only about 4 seconds - how we change to continous ?"),
the Companion and Watch-Together engine may listen continuously to the default audio sink's `.monitor`
(system output only, NEVER a microphone) using `origin_attestation="companion_system_audio"`.

Privacy & Epistemic Contract:
  * System audio is read from `parec` (`@DEFAULT_MONITOR@` / `<default-sink>.monitor`) into a fixed-size
    in-memory circular ring buffer (default 12 seconds at 16 kHz mono s16le = 384 KB) and NEVER written to disk.
  * Every overwritten ring-buffer slot is explicitly zeroized in place (`explicit_zeroize_bytearray`), and
    `.close()` terminates `parec` and zeroizes the entire ring buffer.
  * `ContinuousElectricEarStream` retains a rolling ring of `window` (default 8) Layer A measurement summaries
    (never raw PCM samples, never ASR/lyrics/speech transcription) to track:
      - confidence-weighted `stabilized_bpm` across consecutive windows,
      - cross-window `energy_delta_u8` and `bass_delta_pct`,
      - temporal `ear_regime` (`initializing` | `silence` | `steady` | `building` | `drop` | `breakdown`).
"""
from __future__ import annotations

from collections import deque
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from electric_vision_agent_bridge import ElectricVisionAgentSession
from prod_gate_enclave import explicit_zeroize_bytearray

try:
    from electric_audio_agent_bridge import (
        ElectricAudioAgentSession,
        validate_and_translate_consented_pcm_samples,
    )
except ImportError:
    from electric_vision_agent_bridge import (  # type: ignore[attr-defined]
        ElectricAudioAgentSession,
        validate_and_translate_consented_pcm_samples,
    )

ORIGIN = "companion_system_audio"
SAMPLE_RATE_HZ = 16000
CONSENT_SCOPE = "Companion continuous listening to system audio, measurements only, never saved"


def resolve_default_monitor_source() -> str:
    """Resolves the default PulseAudio/PipeWire output sink's `.monitor` source (never a microphone)."""
    monitor_source = "@DEFAULT_MONITOR@"
    try:
        sink_out = subprocess.run(
            ["/usr/bin/pactl", "get-default-sink"],
            env={"PATH": "/usr/bin:/bin", "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")},
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        sink_name = (sink_out.stdout or "").strip()
        if sink_name and not sink_name.endswith(".monitor"):
            monitor_source = f"{sink_name}.monitor"
    except Exception:
        pass
    return monitor_source


class ContinuousSystemAudioMonitor:
    """
    Persistent in-memory sliding-window PCM ring buffer backed by a single `parec`
    process attached to the default audio sink `.monitor` (never a microphone).

    - Keeps up to `ring_seconds` (default 12.0s) of 16 kHz mono s16le audio in fixed
      0.25s bytearray slots.
    - Explicitly zeroizes every evicted slot on overwrite and zeroizes all slots on `.close()`.
    - Once warm (`>= min_ready_s`), `read_window_samples()` returns the latest `duration_s`
      window in < 5 ms without blocking or dropping beats between calls.
    """

    def __init__(
        self,
        *,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
        ring_seconds: float = 12.0,
        slot_seconds: float = 0.25,
    ) -> None:
        self.sample_rate_hz = int(sample_rate_hz)
        self.slot_seconds = max(0.1, float(slot_seconds))
        self.max_slots = max(8, int(round(float(ring_seconds) / self.slot_seconds)))
        self.slot_samples = max(160, int(round(self.sample_rate_hz * self.slot_seconds)))
        self.slot_bytes = self.slot_samples * 2

        self._lock = threading.Lock()
        self._slots: Deque[bytearray] = deque()
        self._total_chunks_ingested = 0
        self._closed = False
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._thread: Optional[threading.Thread] = None
        self.monitor_source = resolve_default_monitor_source()
        self._start_reader()

    def _start_reader(self) -> None:
        env = {
            "PATH": "/usr/bin:/bin",
            "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"),
        }
        for k in ("PULSE_SERVER", "PIPEWIRE_REMOTE", "HOME", "USER"):
            if k in os.environ:
                env[k] = os.environ[k]
        try:
            self._proc = subprocess.Popen(
                [
                    "/usr/bin/parec",
                    "--format=s16le",
                    f"--rate={self.sample_rate_hz}",
                    "--channels=1",
                    f"--device={self.monitor_source}",
                    "--latency-msec=50",
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            self._proc = None
            return

        self._thread = threading.Thread(
            target=self._reader_loop,
            name="cyclops-continuous-ear-parec",
            daemon=True,
        )
        self._thread.start()

    def _reader_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        pending = bytearray()
        try:
            while not self._closed:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                pending.extend(chunk)
                while len(pending) >= self.slot_bytes:
                    slot_buf = bytearray(pending[: self.slot_bytes])
                    del pending[: self.slot_bytes]
                    with self._lock:
                        if self._closed:
                            explicit_zeroize_bytearray(slot_buf)
                            break
                        while len(self._slots) >= self.max_slots:
                            evicted = self._slots.popleft()
                            explicit_zeroize_bytearray(evicted)
                        self._slots.append(slot_buf)
                        self._total_chunks_ingested += 1
        except Exception:
            pass
        finally:
            explicit_zeroize_bytearray(pending)

    @property
    def buffered_seconds(self) -> float:
        with self._lock:
            return round(len(self._slots) * self.slot_seconds, 2)

    def read_window_samples(
        self,
        *,
        duration_s: float = 4.0,
        min_ready_s: float = 1.5,
        max_wait_s: float = 2.0,
    ) -> List[int]:
        """
        Returns the most recent `duration_s` of 16-bit signed mono PCM samples from the
        live sliding ring buffer, waiting at most `max_wait_s` on initial cold start for
        `min_ready_s` of audio to arrive. Once warm, returns immediately (< 5 ms).
        """
        target_s = max(0.5, min(12.0, float(duration_s)))
        ready_s = max(0.25, min(target_s, float(min_ready_s)))
        deadline = time.monotonic() + max(0.0, float(max_wait_s))

        while time.monotonic() < deadline:
            if self.buffered_seconds >= ready_s or self._proc is None or self._closed:
                break
            time.sleep(0.05)

        needed_slots = max(1, int(round(target_s / self.slot_seconds)))
        combined = bytearray()
        try:
            with self._lock:
                selected = list(self._slots)[-needed_slots:]
                for s in selected:
                    combined.extend(s)
            n_samples = len(combined) // 2
            if n_samples < 64:
                return [0] * max(64, int(round(self.sample_rate_hz * min(1.0, target_s))))
            return list(struct.unpack(f"<{n_samples}h", combined[: n_samples * 2]))
        finally:
            explicit_zeroize_bytearray(combined)

    def close(self) -> None:
        self._closed = True
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=0.8)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass
        with self._lock:
            while self._slots:
                slot = self._slots.popleft()
                explicit_zeroize_bytearray(slot)


class ContinuousElectricEarStream:
    """
    Rolling Layer A temporal memory for the Electric Ear (analogous to `ContinuousElectricEyeStream`).
    Keeps the last `window_frames` (default 8) window summaries (never raw PCM) to compute:
      - confidence-weighted `stabilized_bpm` across consecutive windows,
      - cross-window `transition_from_previous` (`energy_delta_u8`, `mean_energy_delta_u8`, `bass_delta_pct`),
      - temporal `ear_regime` (`initializing` | `silence` | `steady` | `building` | `drop` | `breakdown`).
    """

    def __init__(self, *, window_frames: int = 8) -> None:
        self.window_frames = max(2, int(window_frames))
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.window_frames)
        self._tick = 0

    @property
    def history_count(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._tick = 0

    def ingest_window_metrics(
        self,
        *,
        t_ms: int,
        raw_bpm: float,
        tempo_confidence: float,
        long_window_bpm: Optional[float] = None,
        long_window_confidence: Optional[float] = None,
        q_fast: int,
        q_slow: int,
        bass_share_pct: float,
        low_mid_share_pct: float,
        vocal_share_pct: float,
        brilliance_share_pct: float,
        section_energy_8: Sequence[int],
        section_drop_detected: bool,
        max_section_drop_delta_u8: int,
        pushed_up_active_bytes: int,
        audio_replay_digest: str,
        observation_receipt_id: str,
    ) -> Dict[str, Any]:
        self._tick += 1
        prev = self._history[-1] if self._history else None

        # Choose candidate BPM for this tick (prefer longer rolling window if confidence is higher)
        cand_bpm = float(raw_bpm)
        cand_conf = float(tempo_confidence)
        if (
            long_window_bpm is not None
            and long_window_confidence is not None
            and long_window_bpm > 0.0
            and (long_window_confidence >= cand_conf * 0.9 or cand_conf < 0.26)
        ):
            cand_bpm = float(long_window_bpm)
            cand_conf = max(cand_conf, float(long_window_confidence))

        # Compute confidence-weighted stabilized BPM across active recent windows
        active_bpms: List[Tuple[float, float]] = []
        for h in self._history:
            if h["q_fast"] >= 18 and h["cand_bpm"] > 0.0:
                # Fold octave/half-time outliers toward cand_bpm if within 8%
                b = float(h["cand_bpm"])
                w = max(0.05, float(h["cand_conf"]))
                active_bpms.append((b, w))
        if q_fast >= 12 and cand_bpm > 0.0:
            active_bpms.append((cand_bpm, max(0.10, cand_conf * 1.35)))

        if not active_bpms:
            stabilized_bpm = round(cand_bpm, 1)
        else:
            # Anchor around highest-weight BPM in window and average compatible readings (within 15%)
            anchor_bpm = max(active_bpms, key=lambda x: x[1])[0]
            num = 0.0
            den = 0.0
            for b, w in active_bpms:
                adj_b = b
                if anchor_bpm > 0:
                    if abs(b * 2.0 - anchor_bpm) / anchor_bpm < 0.08:
                        adj_b = b * 2.0
                    elif abs(b * 0.5 - anchor_bpm) / anchor_bpm < 0.08:
                        adj_b = b * 0.5
                if anchor_bpm <= 0 or abs(adj_b - anchor_bpm) / anchor_bpm <= 0.16:
                    num += adj_b * w
                    den += w
            stabilized_bpm = round(num / den, 1) if den > 0 else round(cand_bpm, 1)

        # Cross-window transition metrics
        if prev is None:
            energy_delta_u8 = None
            mean_energy_delta_u8 = None
            bass_delta_pct = None
            vocal_delta_pct = None
        else:
            energy_delta_u8 = int(q_fast) - int(prev["q_fast"])
            mean_energy_delta_u8 = int(q_slow) - int(prev["q_slow"])
            bass_delta_pct = round(float(bass_share_pct) - float(prev["bass_share_pct"]), 1)
            vocal_delta_pct = round(float(vocal_share_pct) - float(prev["vocal_share_pct"]), 1)

        # Intra-window section build check (last 3 segments vs first 3 segments)
        sec = list(section_energy_8) if section_energy_8 else [0] * 8
        first_half = sum(sec[:4]) / max(1, len(sec[:4]))
        second_half = sum(sec[4:]) / max(1, len(sec[4:]))
        intra_build_delta = round(second_half - first_half, 1)

        # Determine temporal ear_regime
        if q_fast < 12 and pushed_up_active_bytes == 0:
            ear_regime = "silence"
        elif prev is None:
            ear_regime = "drop" if section_drop_detected else ("building" if intra_build_delta >= 18.0 else "initializing")
        elif section_drop_detected or (
            (energy_delta_u8 is not None and energy_delta_u8 >= 36)
            and (bass_share_pct >= 30.0 or (bass_delta_pct is not None and bass_delta_pct >= 10.0))
        ):
            ear_regime = "drop"
        elif (mean_energy_delta_u8 is not None and mean_energy_delta_u8 >= 12) or intra_build_delta >= 18.0:
            ear_regime = "building"
        elif (
            (mean_energy_delta_u8 is not None and mean_energy_delta_u8 <= -14)
            or (bass_delta_pct is not None and bass_delta_pct <= -10.0 and vocal_share_pct >= 20.0)
        ):
            ear_regime = "breakdown"
        else:
            ear_regime = "steady"

        entry = {
            "tick": self._tick,
            "t_ms": int(t_ms),
            "raw_bpm": round(float(raw_bpm), 1),
            "cand_bpm": round(cand_bpm, 1),
            "cand_conf": round(cand_conf, 3),
            "stabilized_bpm": stabilized_bpm,
            "q_fast": int(q_fast),
            "q_slow": int(q_slow),
            "bass_share_pct": float(bass_share_pct),
            "low_mid_share_pct": float(low_mid_share_pct),
            "vocal_share_pct": float(vocal_share_pct),
            "brilliance_share_pct": float(brilliance_share_pct),
            "ear_regime": ear_regime,
            "energy_delta_u8": energy_delta_u8,
            "mean_energy_delta_u8": mean_energy_delta_u8,
            "bass_delta_pct": bass_delta_pct,
            "vocal_delta_pct": vocal_delta_pct,
            "intra_build_delta": intra_build_delta,
            "audio_replay_digest": audio_replay_digest,
            "observation_receipt_id": observation_receipt_id,
        }
        self._history.append(entry)
        return {
            "tick": self._tick,
            "window_count": len(self._history),
            "stabilized_bpm": stabilized_bpm,
            "raw_window_bpm": round(float(raw_bpm), 1),
            "tempo_confidence": round(cand_conf, 3),
            "ear_regime": ear_regime,
            "transition_from_previous": {
                "energy_delta_u8": energy_delta_u8,
                "mean_energy_delta_u8": mean_energy_delta_u8,
                "bass_delta_pct": bass_delta_pct,
                "vocal_delta_pct": vocal_delta_pct,
                "intra_build_delta": intra_build_delta,
            },
        }


class DesktopEar:
    """
    A long-lived Companion Ear (`DesktopEar().listen()` / `.close()`), matching `DesktopSight`.
    Maintains a persistent `ContinuousSystemAudioMonitor` (12s sliding `parec` ring buffer)
    and an 8-window `ContinuousElectricEarStream` so every `.listen()` after cold start
    completes in ~25 ms with continuous cross-window memory and stabilized BPM.
    """

    def __init__(self, window: int = 8, ring_seconds: float = 12.0) -> None:
        self._run = tempfile.mkdtemp(prefix="cea-", dir="/tmp")
        os.chmod(self._run, 0o700)
        self._vision_session = ElectricVisionAgentSession(run_dir=self._run, default_resolution=(512, 512))
        self._vision_session.__enter__()
        self._audio_session = ElectricAudioAgentSession(
            run_dir=self._run,
            default_grid=(256, 64),
            shared_vision_session=self._vision_session,
        )
        self._audio_session.__enter__()
        self._monitor = ContinuousSystemAudioMonitor(sample_rate_hz=SAMPLE_RATE_HZ, ring_seconds=ring_seconds)
        self._stream = ContinuousElectricEarStream(window_frames=window)
        self._tick = 0

    def listen(self, *, window_seconds: float = 4.0) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self._tick += 1
        ver = f"v{self._tick}"

        pcm_short = self._monitor.read_window_samples(
            duration_s=window_seconds,
            min_ready_s=min(1.5, window_seconds),
            max_wait_s=2.0,
        )
        tier2 = validate_and_translate_consented_pcm_samples(
            pcm_short,
            sample_rate_hz=SAMPLE_RATE_HZ,
            origin_attestation=ORIGIN,
            consent_token={
                "consented": True,
                "token_id": f"tok_cear_{self._tick:04d}",
                "scope": CONSENT_SCOPE,
            },
            time_slots=256,
            freq_bands=64,
        )

        long_bpm: Optional[float] = None
        long_conf: Optional[float] = None
        if self._monitor.buffered_seconds >= max(6.0, window_seconds + 2.0):
            pcm_long = self._monitor.read_window_samples(
                duration_s=min(10.0, self._monitor.buffered_seconds),
                min_ready_s=6.0,
                max_wait_s=0.05,
            )
            t2_long = validate_and_translate_consented_pcm_samples(
                pcm_long,
                sample_rate_hz=SAMPLE_RATE_HZ,
                origin_attestation=ORIGIN,
                consent_token={
                    "consented": True,
                    "token_id": f"tok_cear_long_{self._tick:04d}",
                    "scope": CONSENT_SCOPE,
                },
                time_slots=256,
                freq_bands=64,
            )
            dt_l = t2_long["dynamic_and_tempo_profile"]
            long_bpm = round(dt_l["estimated_bpm_x10"] / 10.0, 1)
            long_conf = round(dt_l["tempo_confidence_q16"] / 65536.0, 3)

        self._audio_session.register_chargeable_audio(
            "src_companion_ear",
            ver,
            tier2["translated_events"],
            time_slots=256,
            freq_bands=64,
            playhead_ratio=1.0,
            origin_attestation=ORIGIN,
        )
        aud_insp = self._audio_session.inspect_audio(
            "src_companion_ear",
            ver,
            issue_observation_receipt=True,
        )

        fb = tier2["four_band_polyphonic_profile"]
        dt = tier2["dynamic_and_tempo_profile"]
        cm = aud_insp["charge_metrics"]
        sum_prof = cm["op_charge_summary_u8"]
        push_prof = cm["op_push_up_profile_q16"]

        raw_bpm = round(dt["estimated_bpm_x10"] / 10.0, 1)
        tempo_conf = round(dt["tempo_confidence_q16"] / 65536.0, 3)
        bass_share = round((fb["sub_bass_55_280hz_share_q16"] * 100.0) / 65536.0, 1)
        low_mid_share = round((fb["low_mid_280_900hz_share_q16"] * 100.0) / 65536.0, 1)
        vocal_share = round((fb["vocal_formant_900_3400hz_share_q16"] * 100.0) / 65536.0, 1)
        brill_share = round((fb["high_brilliance_3400_7040hz_share_q16"] * 100.0) / 65536.0, 1)
        q_fast = max(int(dt["peak_slot_energy_u8"]), min(255, int(push_prof["max_charge_u8"])))
        q_slow = max(int(dt["mean_slot_energy_u8"]), min(255, int(sum_prof["mean_slow_charge_q16"] >> 16) * 4))

        cont = self._stream.ingest_window_metrics(
            t_ms=int(time.time() * 1000),
            raw_bpm=raw_bpm,
            tempo_confidence=tempo_conf,
            long_window_bpm=long_bpm,
            long_window_confidence=long_conf,
            q_fast=q_fast,
            q_slow=q_slow,
            bass_share_pct=bass_share,
            low_mid_share_pct=low_mid_share,
            vocal_share_pct=vocal_share,
            brilliance_share_pct=brill_share,
            section_energy_8=dt["section_envelope_8seg_u8"],
            section_drop_detected=dt["section_drop_detected"],
            max_section_drop_delta_u8=dt["max_section_drop_delta_u8"],
            pushed_up_active_bytes=sum_prof["pushed_up_active_bytes"],
            audio_replay_digest=aud_insp["audio_replay_digest"],
            observation_receipt_id=aud_insp["render"]["observation_receipt_id"],
        )

        bpm = cont["stabilized_bpm"]
        regime = cont["ear_regime"]
        if regime == "silence":
            words = "quiet / near silence"
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
            words = f"{tone_char} at {bpm:.1f} BPM (bass {bass_share:.1f}%, vocal-band {vocal_share:.1f}%, regime={regime})"
            ear_line = f"Ear: {tone_char}, {bpm:.1f} BPM, q_fast={q_fast}/q_slow={q_slow}, {regime}"

        return {
            "at": int(time.time() * 1000),
            "origin": ORIGIN,
            "earLine": ear_line,
            "words": words,
            "measures": {
                "stabilized_bpm": bpm,
                "raw_window_bpm": raw_bpm,
                "tempo_confidence": cont["tempo_confidence"],
                "ear_regime": regime,
                "q_fast": q_fast,
                "q_slow": q_slow,
                "bass_share_pct": bass_share,
                "low_mid_share_pct": low_mid_share,
                "vocal_share_pct": vocal_share,
                "brilliance_share_pct": brill_share,
                "section_energy_8": dt["section_envelope_8seg_u8"],
                "transition_from_previous": cont["transition_from_previous"],
                "buffered_seconds": self._monitor.buffered_seconds,
                "window_count": cont["window_count"],
            },
            "receipt": {
                "observation_receipt_id": aud_insp["render"]["observation_receipt_id"],
                "audio_replay_digest": aud_insp["audio_replay_digest"],
                "tick": self._tick,
            },
            "elapsed_ms": int(round((time.perf_counter() - t0) * 1000.0)),
        }

    def close(self) -> None:
        try:
            self._monitor.close()
        finally:
            try:
                self._audio_session.__exit__(None, None, None)
            finally:
                try:
                    self._vision_session.__exit__(None, None, None)
                finally:
                    shutil.rmtree(self._run, ignore_errors=True)
