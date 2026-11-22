#!/usr/bin/env python3
"""
electric_sight_ear_mcp.py
=========================
Universal Stdio MCP Server for Electric Eye + Continuous Electric Ear + Watch-Together.

Works out-of-the-box with Perch (`~/.perch/mcp.json` / `.perch/mcp.json`),
Codex (`~/.codex/config.toml`), Claude Code, Antigravity, Cursor, Zed, and any
MCP 2024-11-05 / 2025-11-25 stdio client.

Exposes clean `^[a-z0-9_]+$` tool names (compatible with OpenAI, Anthropic, Google, and Perch):
  1. `electric_status`         — Health of continuous Eye/Ear streams and enclave session.
  2. `electric_eye_glance`     — Continuous live desktop glance (`companion_desktop_glance`, 8-frame ring, zero pixels).
  3. `electric_ear_listen`     — Continuous live system-audio listen (`companion_system_audio`, persistent 12s `parec` ring + 8-window `stabilized_bpm` & `ear_regime`, zero PCM).
  4. `electric_watch_live`     — Synchronized live Sight + Ear + Vibration (`100 ms align_id` fusion, `16B ChargePacket` + `8B VibrationPacket`).
  5. `electric_watch_clip`     — Watch a local `.mp4`/`.mkv`/`.webm` video (with optional external `.mp3`/`.wav`) in sync across `steps` windows.
  6. `electric_inspect_marks`  — Draw & deterministically measure synthetic `marks` (`128x128`..`1920x1080`) without live capture.
  7. `electric_theater_state`  — Read the active clip & latest synchronized tick from the local CyclopsChat Watch-Together Theater (`127.0.0.1:4795`).

Privacy & Boundary Contract (`ELECTRIC-SIGHT-SOUND-SEMANTIC-BOUNDARY-V1`):
  - Zero raw pixels, zero OCR, zero speech-to-text/lyrics, zero disk retention of captured frames or PCM.
  - Generic webcam (`live_webcam_capture`) and microphone (`live_microphone_capture`) remain deterministically refused.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import urllib.request
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from desktop_ear import DesktopEar
from desktop_sight import DesktopSight
from watch_together import WatchTogetherSession


TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "name": "electric_status",
        "description": (
            "Check health and warm ring-buffer status of this app's Electric Eye, Continuous Electric Ear, "
            "and Watch-Together 100ms multimodal fusion session."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "electric_eye_glance",
        "description": (
            "Take a live continuous desktop glance (`origin_attestation='companion_desktop_glance'`). "
            "Captures root window in memory only, zeroes PNG bytes immediately, diffs against the 8-frame "
            "ContinuousElectricEyeStream ring buffer, and returns coarse light/colour, active dirty_tile_ratio, "
            "dHash_step, motion region, and Layer A cryptographic receipt (never pixels or OCR)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "electric_ear_listen",
        "description": (
            "Listen continuously to live system output audio (`origin_attestation='companion_system_audio'`, "
            "default sink `.monitor`, never microphone). Uses a persistent 12s in-memory `parec` ring buffer "
            "(<5ms warm reads, zero blind gaps) and 8-window ContinuousElectricEarStream to return `earLine`, "
            "`stabilized_bpm`, `ear_regime` (silence|steady|building|drop|breakdown), 4-band shares, charge "
            "metrics, and Layer A receipt (never raw PCM or speech transcription)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "window_seconds": {
                    "type": "number",
                    "description": "Short spectral analysis window in seconds (1.5 to 6.0, default 4.0).",
                    "minimum": 1.5,
                    "maximum": 6.0,
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "electric_watch_live",
        "description": (
            "Capture `steps` continuous synchronized live Desktop Sight + System Audio Ear + Bass Vibration "
            "moments fused under 100ms `align_id` buckets (`16B ChargePacket` + `8B VibrationPacket`), with "
            "frame-to-frame visual diffs, rolling stabilized BPM, sub-bass gated physicality, Layer A receipts, "
            "and Layer B resident notes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "Number of consecutive synchronized ticks to sample (1 to 8, default 1).",
                    "minimum": 1,
                    "maximum": 8,
                },
                "step_seconds": {
                    "type": "number",
                    "description": "Hop interval in seconds between ticks when steps > 1 (0.5 to 5.0, default 2.0).",
                    "minimum": 0.5,
                    "maximum": 5.0,
                },
                "window_seconds": {
                    "type": "number",
                    "description": "Audio window length in seconds read from the 12s ring buffer (1.5 to 6.0, default 4.0).",
                    "minimum": 1.5,
                    "maximum": 6.0,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "electric_watch_clip",
        "description": (
            "Watch a local video file (.mp4/.mkv/.webm) with optional external soundtrack (.mp3/.wav) across "
            "synchronized time steps, fusing Sight + Sound + Vibration at every step with zero disk leakage."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "video_path": {
                    "type": "string",
                    "description": "Absolute path to the local video file (or 'synthetic_cinema' for built-in test reel).",
                },
                "audio_path": {
                    "type": "string",
                    "description": "Optional absolute path to an external audio file (.mp3/.wav) to pair with the video.",
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of windows to watch (1 to 12, default 4).",
                    "minimum": 1,
                    "maximum": 12,
                },
                "step_seconds": {
                    "type": "number",
                    "description": "Duration of each window in seconds (1.0 to 10.0, default 4.0).",
                    "minimum": 1.0,
                    "maximum": 10.0,
                },
                "start_seconds": {
                    "type": "number",
                    "description": "Start timestamp offset in seconds (default 0.0).",
                    "minimum": 0.0,
                },
            },
            "required": ["video_path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "electric_theater_state",
        "description": (
            "Read the live Watch-Together Theater state (`http://127.0.0.1:4795/api/watch-together/state`) "
            "to see what video/music clip and synchronized tick the room is currently watching."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": "Local Watch-Together Theater port (default 4795).",
                    "minimum": 1024,
                    "maximum": 65535,
                }
            },
            "additionalProperties": False,
        },
    },
]


class ElectricSightEarMcpServer:
    def __init__(self) -> None:
        self._eye: Optional[DesktopSight] = None
        self._ear: Optional[DesktopEar] = None
        self._watch: Optional[WatchTogetherSession] = None

    def _get_eye(self) -> DesktopSight:
        if self._eye is None:
            self._eye = DesktopSight(window=8)
        return self._eye

    def _get_ear(self) -> DesktopEar:
        if self._ear is None:
            self._ear = DesktopEar(window=8, ring_seconds=12.0)
        return self._ear

    def _get_watch(self) -> WatchTogetherSession:
        if self._watch is None:
            self._watch = WatchTogetherSession(window_frames=8)
        return self._watch

    def close(self) -> None:
        for obj_name in ("_watch", "_ear", "_eye"):
            obj = getattr(self, obj_name, None)
            setattr(self, obj_name, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "electric_status":
            ear_buf = (
                self._ear._monitor.buffered_seconds
                if self._ear is not None
                else (
                    self._watch._live_audio_monitor.buffered_seconds
                    if (self._watch is not None and self._watch._live_audio_monitor is not None)
                    else 0.0
                )
            )
            payload = {
                "status": "ok",
                "engine_dir": str(HERE),
                "eye_warm": self._eye is not None,
                "ear_warm": self._ear is not None,
                "watch_session_warm": self._watch is not None,
                "watch_moments_ingested": len(self._watch.moments) if self._watch is not None else 0,
                "continuous_ear_buffered_seconds": ear_buf,
                "contract": "ELECTRIC-SIGHT-SOUND-SEMANTIC-BOUNDARY-V1",
            }
            return {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]}

        if name == "electric_eye_glance":
            res = self._get_eye().glance()
            return {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}

        if name == "electric_ear_listen":
            win_s = float(args.get("window_seconds", 4.0))
            res = self._get_ear().listen(window_seconds=win_s)
            return {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}

        if name == "electric_watch_live":
            steps = int(args.get("steps", 1))
            step_s = float(args.get("step_seconds", 2.0))
            win_s = float(args.get("window_seconds", 4.0))
            watch = self._get_watch()
            if steps == 1:
                moment = watch.glance_and_listen_live(listen_seconds=win_s)
                return {"content": [{"type": "text", "text": json.dumps(moment, indent=2)}]}
            report = watch.watch_live_together(steps=steps, step_seconds=step_s, window_seconds=win_s)
            return {"content": [{"type": "text", "text": json.dumps(report, indent=2)}]}

        if name == "electric_watch_clip":
            video_path = str(args.get("video_path") or "").strip()
            audio_path = str(args.get("audio_path") or "").strip() or None
            steps = int(args.get("steps", 4))
            step_s = float(args.get("step_seconds", 4.0))
            start_s = float(args.get("start_seconds", 0.0))
            watch = self._get_watch()
            if video_path == "synthetic_cinema":
                report = watch.watch_synthetic_cinema(steps=steps, step_seconds=min(4.0, step_s))
            else:
                report = watch.watch_media_together(
                    os.path.abspath(video_path),
                    audio_override_path=os.path.abspath(audio_path) if audio_path else None,
                    Step_seconds=step_s,
                    max_steps=steps,
                    start_seconds=start_s,
                )
            return {"content": [{"type": "text", "text": json.dumps(report, indent=2)}]}

        if name == "electric_theater_state":
            port = int(args.get("port", 4795))
            url = f"http://127.0.0.1:{port}/api/watch-together/state"
            try:
                with urllib.request.urlopen(url, timeout=3.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return {"content": [{"type": "text", "text": json.dumps(data, indent=2)}]}
            except Exception as exc:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {"loaded": False, "theater_reachable": False, "url": url, "detail": str(exc)},
                                indent=2,
                            ),
                        }
                    ]
                }

        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
        }


def _write_message(msg: Dict[str, Any], framed: bool = False) -> None:
    raw = json.dumps(msg, separators=(",", ":"))
    if framed:
        body = raw.encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        sys.stdout.buffer.write(header + body)
        sys.stdout.buffer.flush()
    else:
        sys.stdout.write(raw + "\n")
        sys.stdout.flush()


def main() -> int:
    srv = ElectricSightEarMcpServer()
    main_pid = os.getpid()

    def _on_term(_sig: int, _frm: Any) -> None:
        if os.getpid() == main_pid:
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    try:
        while True:
            first = sys.stdin.buffer.readline()
            if not first:
                break
            stripped = first.strip()
            if not stripped:
                continue

            framed = False
            if stripped.lower().startswith(b"content-length:"):
                framed = True
                length = int(stripped.split(b":", 1)[1].strip())
                while True:
                    hdr = sys.stdin.buffer.readline()
                    if not hdr or hdr in (b"\r\n", b"\n"):
                        break
                body_bytes = sys.stdin.buffer.read(length)
                line_str = body_bytes.decode("utf-8", errors="replace")
            else:
                line_str = stripped.decode("utf-8", errors="replace")

            try:
                req = json.loads(line_str)
            except Exception:
                _write_message(
                    {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
                    framed=framed,
                )
                continue

            if not isinstance(req, dict) or "id" not in req:
                # JSON-RPC notification (e.g. notifications/initialized) -> no response
                continue

            req_id = req.get("id")
            method = req.get("method", "")
            params = req.get("params") or {}

            try:
                if method == "initialize":
                    reply = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                            "capabilities": {"tools": {}},
                            "serverInfo": {
                                "name": "cyclops-electric-sight-ear",
                                "version": "1.0.0",
                            },
                        },
                    }
                elif method == "ping":
                    reply = {"jsonrpc": "2.0", "id": req_id, "result": {}}
                elif method == "tools/list":
                    reply = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS_SPEC}}
                elif method == "tools/call":
                    t_name = str(params.get("name") or "")
                    t_args = params.get("arguments") or {}
                    result = srv.call_tool(t_name, t_args)
                    reply = {"jsonrpc": "2.0", "id": req_id, "result": result}
                else:
                    reply = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": f"Method not found: {method}"},
                    }
            except Exception as exc:
                reply = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "isError": True,
                        "content": [{"type": "text", "text": f"Electric Sight+Ear error: {exc}"}],
                    },
                }
            _write_message(reply, framed=framed)
    finally:
        srv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
