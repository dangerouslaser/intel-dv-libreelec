#!/usr/bin/env python3
"""Time a requested Kodi channel change; emit only sanitized timing events.

Requires Kodi TCP JSON-RPC (9090), SSH, and debug logging without the overlay.
SSH uses sshpass with SSHPASS if supplied, otherwise normal SSH authentication.
Player notifications are software milestones, not measurements of visible pixels.
"""
import argparse
import datetime
import json
import os
import re
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("channel", type=int)
    parser.add_argument("--seconds", type=float, default=18)
    args = parser.parse_args()
    ssh = (["sshpass", "-e"] if os.environ.get("SSHPASS") else []) + [
        "ssh", "-o", "ConnectTimeout=5", "root@" + args.host]
    log = "/storage/.kodi/temp/kodi.log"
    initial = subprocess.check_output(
        ssh + [f"wc -c < {log}; cut -d ' ' -f1 /proc/uptime"], text=True).splitlines()
    offset, device_start_ms = int(initial[0]), float(initial[1]) * 1000
    events = []
    decoder = json.JSONDecoder()
    with socket.create_connection((args.host, 9090), timeout=5) as connection:
        connection.settimeout(0.25)
        started = time.monotonic()
        connection.sendall(json.dumps({"jsonrpc": "2.0", "id": 1,
            "method": "Player.Open", "params": {"item": {"channelid": args.channel}}}
        ).encode() + b"\n")
        pending = ""
        while time.monotonic() - started < args.seconds:
            try:
                chunk = connection.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                raise RuntimeError("Kodi closed its notification connection")
            pending += chunk.decode("utf-8")
            while pending.strip():
                pending = pending.lstrip()
                try:
                    message, consumed = decoder.raw_decode(pending)
                except json.JSONDecodeError:
                    break
                pending = pending[consumed:]
                method = message.get("method", "")
                if message.get("id") == 1:
                    if "error" in message:
                        raise RuntimeError("Kodi rejected Player.Open")
                    method = "Player.Open.reply"
                if method.startswith("Player."):
                    events.append({"event": method,
                                   "request_ms": round((time.monotonic() - started) * 1000, 1)})

    raw = subprocess.check_output(ssh + [f"tail -c +{offset + 1} {log}"], text=True)
    patterns = [
        ("input_open", "Creating InputStream"),
        ("demux_open", "Creating Demuxer"),
        ("http_open", "CCurlFile::Open"),
        ("probe_start", "avformat_find_stream_info starting"),
        ("probe_end", "av_find_stream_info finished"),
        ("missing_video_headers", "require extradata"),
        ("video_codec", "Creating video codec:"),
        ("audio_thread", "Creating audio thread"),
        ("audio_stream", "Creating audio stream"),
        ("decoder_fps", "Live startup: decoder supplied framerate"),
        ("renderer_config", "CRenderManager::Configure - change configuration"),
        ("rate_correction", "CalcFrameRate framerate was:"),
        ("refresh_select", "Display resolution ADJUST"),
        ("display_lost", "VideoPlayer: OnLostDisplay received"),
        ("display_reset", "VideoPlayer: OnResetDisplay received"),
        ("display_commit", "DVBridge signal committed:"),
        ("audio_resync", "CVideoPlayerAudio - CDVDMsg::GENERAL_RESYNC"),
        ("video_resync", "CVideoPlayerVideo - CDVDMsg::GENERAL_RESYNC"),
        ("av_started", "CApplicationPlayerCallback::OnAVStarted"),
        ("cache_state", "CVideoPlayer::SetCaching - caching state"),
        ("audio_start_timing", "VideoPlayer::Sync - Audio - pts:"),
        ("video_start_timing", "VideoPlayer::Sync - Video - pts:"),
        ("audio_sync_start", "ActiveAE - start sync of audio stream"),
        ("audio_sync_adjust", "ActiveAE::SyncStream - average error"),
        ("audio_sink_ready", "OpenSink(): ALSA Initialized:"),
        ("stream_stalled", "Stream stalled, start buffering"),
    ]
    timeline = []
    anchor = None
    for line in raw.splitlines():
        match = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)", line)
        if not match:
            continue
        for name, pattern in patterns:
            if pattern not in line:
                continue
            stamp = datetime.datetime.fromisoformat(match[1])
            if anchor is None:
                anchor = stamp
            entry = {"event": name, "device_time": match[1],
                     "log_ms": round((stamp - anchor).total_seconds() * 1000, 1)}
            if name in ("audio_start_timing", "video_start_timing"):
                for key, value in re.findall(r"(pts|cache|totalcache): ([-\d.]+)", line):
                    entry[key + "_us"] = float(value)
            if name == "audio_sync_adjust":
                value = re.search(r"average error (?:of )?([-\d.]+)", line)
                if value:
                    entry["error_ms"] = float(value[1])
            if name == "renderer_config":
                rate = re.search(r"framerate: (\d+\.\d+)", line)
                if rate:
                    entry["fps"] = rate[1]
            if name == "display_commit":
                for key in ("dv", "hdr_blob", "colorspace", "clock_khz", "totals"):
                    value = re.search(r"\b" + key + r"=([\dx]+)", line)
                    if value:
                        entry[key] = value[1]
            timeline.append(entry)
    trace = subprocess.check_output(ssh + [
        "journalctl -u kodi -b -o cat --no-pager | grep '^ZAPTRACE' || true"], text=True)
    probe_events = []
    for line in trace.splitlines():
        match = re.search(r"mono_ms=([\d.]+)", line)
        if match and float(match[1]) >= device_start_ms:
            probe_events.append(line)
    print(json.dumps({"channelid": args.channel, "sample_seconds": args.seconds,
                     "request_events": events, "log_events": timeline,
                     "probe_events": probe_events}, indent=2))


if __name__ == "__main__":
    main()
