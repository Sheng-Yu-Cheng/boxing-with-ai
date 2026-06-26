#!/usr/bin/env python3
r"""
radarbox_3tx_beam_static_roi_logger.py

DCA1000 UDP receive tester/logger for RadarBox AWR2243BOOST, with static range-profile ROI metrics for beam verification.

Target setup:
  AWR2243BOOST + DCA1000
  3Tx simultaneous Tx beamforming, 4Rx receive
  Complex1x 16-bit ADC
  chirp 0 = TX0 + TX1 + TX2 simultaneous
  FrameConfig(0, 0, Nframes, loops, framePeriodMs, 0, 1)

For 3Tx simultaneous beamforming, the three TX are NOT separated in raw ADC.
The receiver still sees 4 RX channels for each chirp:
  frame_bytes = loops * samples * rx * IQ * bytes_per_sample

Outputs:
  run.log       console-style text log
  frames.csv    one row per assembled frame, including current beam command
  status.jsonl  periodic status records
  latest.json   overwritten with latest frame/status
  summary.json  final summary after exit

Extra beam-verification outputs:
  static_peak_* uses range FFT power before slow-time clutter removal.
  static_roi_* measures a fixed range window, which is better for proving Tx beam changes than the global Doppler peak.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import socket
import time
from pathlib import Path
from typing import Optional

import numpy as np


@dataclasses.dataclass
class TestConfig:
    pc_ip: str = "192.168.33.30"
    dca_ip: str = "192.168.33.180"
    data_port: int = 4098
    socket_rcvbuf: int = 64 * 1024 * 1024
    recvfrom_bytes: int = 9000
    dca_header_bytes: int = 10

    loops: int = 64
    tx_groups: int = 1  # 1 for 3Tx simultaneous; 3 only for TDM-MIMO.
    rx: int = 4
    samples: int = 256
    iq: int = 2
    bytes_per_sample: int = 2

    frame_period_ms: float = 20.0
    idle_us: float = 20.0
    ramp_end_us: float = 60.0
    sample_rate_ksps: float = 10000.0
    slope_mhz_us: float = 29.982
    start_freq_ghz: float = 77.0

    frames: int = 8
    infinite: bool = False
    timeout_s: float = 10.0
    report_every_s: float = 1.0
    strict: bool = False
    keep_raw_frames: int = 0

    output_dir: str = ""
    log_file: str = ""
    frame_csv: str = ""
    status_jsonl: str = ""
    latest_json: str = ""
    summary_json: str = ""
    save_npz: Optional[str] = None
    no_console: bool = False

    beam_cmd_file: str = r"C:\temp\radarbox_beam_cmd.txt"

    # Static range-profile metrics for beam verification.
    # If roi_range_m is provided, static_roi_power_db tracks a fixed target range
    # instead of whichever Doppler peak happens to be strongest.
    roi_range_m: float = -1.0
    roi_half_width_m: float = 0.20
    static_min_range_m: float = 0.30
    static_max_range_m: float = 8.00

    @property
    def frame_bytes(self) -> int:
        return self.loops * self.tx_groups * self.samples * self.rx * self.iq * self.bytes_per_sample

    @property
    def expected_words_per_frame(self) -> int:
        return self.frame_bytes // 2

    @property
    def expected_fps(self) -> float:
        return 1000.0 / self.frame_period_ms

    @property
    def chirp_period_s(self) -> float:
        return (self.idle_us + self.ramp_end_us) * 1e-6

    @property
    def wavelength_m(self) -> float:
        return 299_792_458.0 / (self.start_freq_ghz * 1e9)

    @property
    def v_max_mps(self) -> float:
        return self.wavelength_m / (4.0 * self.chirp_period_s)

    @property
    def velocity_resolution_mps(self) -> float:
        return self.wavelength_m / (2.0 * self.loops * self.chirp_period_s)

    @property
    def range_resolution_m(self) -> float:
        bandwidth_hz = self.slope_mhz_us * 1e12 * (self.samples / (self.sample_rate_ksps * 1e3))
        return 299_792_458.0 / (2.0 * bandwidth_hz)

    @property
    def expected_stream_MBps(self) -> float:
        return self.frame_bytes * self.expected_fps / 1e6


class TeeLogger:
    def __init__(self, path: Path, no_console: bool = False):
        self.path = path
        self.no_console = no_console
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass

    def log(self, msg: str = "") -> None:
        ts = time.strftime("%H:%M:%S")
        out = f"[{ts}] {msg}"
        if not self.no_console:
            print(out)
        self.f.write(out + "\n")
        self.f.flush()


def pow_to_db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-12))


def parse_dca_packet(data: bytes, header_bytes: int):
    if len(data) < header_bytes:
        return None
    seq = int.from_bytes(data[0:4], byteorder="little", signed=False)
    byte_count = int.from_bytes(data[4:10], byteorder="little", signed=False)
    return seq, byte_count, data[header_bytes:]


def raw_frame_to_adc(frame_bytes: bytes, cfg: TestConfig) -> np.ndarray:
    raw = np.frombuffer(frame_bytes, dtype=np.int16)
    if raw.size != cfg.expected_words_per_frame:
        raise ValueError(f"raw words={raw.size}, expected={cfg.expected_words_per_frame}")
    data = raw.reshape(cfg.loops * cfg.tx_groups, cfg.samples, cfg.iq, cfg.rx)
    i_data = data[:, :, 0, :].astype(np.float32)
    q_data = data[:, :, 1, :].astype(np.float32)
    adc = i_data + 1j * q_data
    return adc.reshape(cfg.loops, cfg.tx_groups, cfg.samples, cfg.rx)


def make_axes(cfg: TestConfig):
    sample_rate_hz = cfg.sample_rate_ksps * 1e3
    slope_hz_s = cfg.slope_mhz_us * 1e12
    freqs = np.fft.fftfreq(cfg.samples, d=1.0 / sample_rate_hz)[: cfg.samples // 2]
    range_axis = 299_792_458.0 * freqs / (2.0 * slope_hz_s)
    fd = np.fft.fftshift(np.fft.fftfreq(cfg.loops, d=cfg.chirp_period_s))
    velocity_axis = fd * cfg.wavelength_m / 2.0
    return range_axis, velocity_axis


def compute_rd_power(adc: np.ndarray, cfg: TestConfig) -> np.ndarray:
    x = adc.copy()
    x = x - np.mean(x, axis=2, keepdims=True)
    range_win = np.hanning(cfg.samples)[None, None, :, None]
    rfft = np.fft.fft(x * range_win, axis=2)
    rfft = rfft[:, :, : cfg.samples // 2, :]
    rfft = rfft - np.mean(rfft, axis=0, keepdims=True)
    doppler_win = np.hanning(cfg.loops)[:, None, None, None]
    rd = np.fft.fftshift(np.fft.fft(rfft * doppler_win, axis=0), axes=0)
    rd_power = np.sum(np.abs(rd) ** 2, axis=(1, 3)).T
    return rd_power.astype(np.float64)


def read_beam_cmd(path_str: str) -> str:
    if not path_str:
        return ""
    try:
        p = Path(path_str)
        if not p.exists():
            return ""
        return p.read_text(encoding="ascii", errors="ignore").strip()
    except Exception:
        return ""


def parse_beam_cmd(cmd: str) -> dict:
    out = {"beam_cmd": cmd, "beam_theta_deg": None, "beam_tx0_code": None, "beam_tx1_code": None, "beam_tx2_code": None}
    if not cmd:
        return out
    try:
        vals = [v.strip() for v in cmd.split(",") if v.strip()]
        if len(vals) == 1:
            out["beam_theta_deg"] = float(vals[0])
        elif len(vals) >= 3:
            out["beam_tx0_code"] = int(float(vals[0])) & 0x3F
            out["beam_tx1_code"] = int(float(vals[1])) & 0x3F
            out["beam_tx2_code"] = int(float(vals[2])) & 0x3F
    except Exception:
        pass
    return out


def compute_static_range_power(adc: np.ndarray, cfg: TestConfig) -> np.ndarray:
    """Return range power before slow-time clutter removal.

    This is the metric for beam-steering verification with a fixed reflector.
    The Doppler/RD path subtracts the slow-time mean, which suppresses static
    objects and makes beam changes hard to see.
    """
    x = adc.copy()
    x = x - np.mean(x, axis=2, keepdims=True)
    range_win = np.hanning(cfg.samples)[None, None, :, None]
    rfft = np.fft.fft(x * range_win, axis=2)
    rfft = rfft[:, :, : cfg.samples // 2, :]
    return np.sum(np.abs(rfft) ** 2, axis=(0, 1, 3)).astype(np.float64)


def analyze_frame(frame_bytes: bytes, cfg: TestConfig, range_axis: np.ndarray, velocity_axis: np.ndarray) -> dict:
    raw = np.frombuffer(frame_bytes, dtype=np.int16)
    adc = raw_frame_to_adc(frame_bytes, cfg)
    rx_rms = np.sqrt(np.mean(np.abs(adc) ** 2, axis=(0, 1, 2)))
    rx_rms_db = 20.0 * np.log10(np.maximum(rx_rms, 1e-12))
    i_mean_by_rx = np.mean(adc.real, axis=(0, 1, 2))
    q_mean_by_rx = np.mean(adc.imag, axis=(0, 1, 2))
    rd_power = compute_rd_power(adc, cfg)
    rd_db = pow_to_db(rd_power)
    finite = rd_db[np.isfinite(rd_db)]
    peak_r, peak_v = np.unravel_index(int(np.argmax(rd_power)), rd_power.shape)

    static_power = compute_static_range_power(adc, cfg)
    static_db = pow_to_db(static_power)
    static_finite = static_db[np.isfinite(static_db)]
    range_mask = (range_axis >= cfg.static_min_range_m) & (range_axis <= cfg.static_max_range_m)
    if not np.any(range_mask):
        range_mask = np.ones_like(range_axis, dtype=bool)
    masked_indices = np.flatnonzero(range_mask)
    static_peak_i = int(masked_indices[int(np.argmax(static_power[range_mask]))])

    roi_power_db = None
    roi_snr_like_db = None
    roi_bin_count = 0
    if cfg.roi_range_m is not None and cfg.roi_range_m >= 0:
        roi_mask = (np.abs(range_axis - cfg.roi_range_m) <= cfg.roi_half_width_m) & range_mask
        roi_bin_count = int(np.sum(roi_mask))
        if roi_bin_count > 0:
            roi_power = float(np.sum(static_power[roi_mask]))
            roi_power_db = float(pow_to_db(roi_power))
            roi_snr_like_db = float(roi_power_db - np.median(static_finite)) if static_finite.size else None

    rec = {
        "raw_min": int(raw.min()) if raw.size else None,
        "raw_max": int(raw.max()) if raw.size else None,
        "raw_mean": float(raw.astype(np.float64).mean()) if raw.size else None,
        "raw_std": float(raw.astype(np.float64).std()) if raw.size else None,
        "zero_ratio": float(np.mean(raw == 0)),
        "saturation_ratio": float(np.mean((raw == np.iinfo(np.int16).min) | (raw == np.iinfo(np.int16).max))),
        "rx_rms_db_span": float(np.max(rx_rms_db) - np.min(rx_rms_db)),
        "rd_peak_range_m": float(range_axis[peak_r]),
        "rd_peak_velocity_mps": float(velocity_axis[peak_v]),
        "rd_peak_power_db": float(rd_db[peak_r, peak_v]),
        "rd_median_power_db": float(np.median(finite)) if finite.size else None,
        "rd_peak_snr_like_db": float(rd_db[peak_r, peak_v] - np.median(finite)) if finite.size else None,
        "static_peak_range_m": float(range_axis[static_peak_i]),
        "static_peak_power_db": float(static_db[static_peak_i]),
        "static_median_power_db": float(np.median(static_finite)) if static_finite.size else None,
        "static_peak_snr_like_db": float(static_db[static_peak_i] - np.median(static_finite)) if static_finite.size else None,
        "static_roi_range_m": float(cfg.roi_range_m) if cfg.roi_range_m is not None and cfg.roi_range_m >= 0 else None,
        "static_roi_half_width_m": float(cfg.roi_half_width_m),
        "static_roi_bin_count": roi_bin_count,
        "static_roi_power_db": roi_power_db,
        "static_roi_snr_like_db": roi_snr_like_db,
    }
    for i in range(cfg.rx):
        rec[f"rx{i}_rms_db"] = float(rx_rms_db[i])
        rec[f"rx{i}_i_mean"] = float(i_mean_by_rx[i])
        rec[f"rx{i}_q_mean"] = float(q_mean_by_rx[i])
    return rec


def default_output_dir() -> str:
    return f"radarbox_beam_test_{time.strftime('%Y%m%d_%H%M%S')}"


def resolve_output_paths(cfg: TestConfig) -> TestConfig:
    if not cfg.output_dir:
        cfg.output_dir = default_output_dir()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.log_file = cfg.log_file or str(out / "run.log")
    cfg.frame_csv = cfg.frame_csv or str(out / "frames.csv")
    cfg.status_jsonl = cfg.status_jsonl or str(out / "status.jsonl")
    cfg.latest_json = cfg.latest_json or str(out / "latest.json")
    cfg.summary_json = cfg.summary_json or str(out / "summary.json")
    return cfg


FRAME_CSV_FIELDS = [
    "frame_id", "timestamp_perf", "elapsed_s", "valid", "invalid_bytes", "reason",
    "beam_cmd", "beam_theta_deg", "beam_tx0_code", "beam_tx1_code", "beam_tx2_code",
    "raw_std", "zero_ratio", "saturation_ratio",
    "rx0_rms_db", "rx1_rms_db", "rx2_rms_db", "rx3_rms_db", "rx_rms_db_span",
    "rd_peak_range_m", "rd_peak_velocity_mps", "rd_peak_power_db", "rd_median_power_db", "rd_peak_snr_like_db",
    "static_peak_range_m", "static_peak_power_db", "static_median_power_db", "static_peak_snr_like_db",
    "static_roi_range_m", "static_roi_half_width_m", "static_roi_bin_count", "static_roi_power_db", "static_roi_snr_like_db",
    "packet_gaps_total", "missing_bytes_total", "invalid_frames_total",
]


def write_json_atomic(path: str, payload: dict):
    try:
        p = Path(path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass


def append_jsonl(path: str, payload: dict):
    try:
        with Path(path).open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


def csv_row(record: dict) -> dict:
    return {k: record.get(k, "") for k in FRAME_CSV_FIELDS}


def print_config(cfg: TestConfig, log: TeeLogger):
    log.log("=== RadarBox 3Tx simultaneous beam receive file logger ===")
    log.log(f"bind              : {cfg.pc_ip}:{cfg.data_port}")
    log.log(f"expected DCA IP   : {cfg.dca_ip}")
    log.log(f"shape             : loops={cfg.loops}, tx_groups={cfg.tx_groups}, rx={cfg.rx}, samples={cfg.samples}")
    log.log(f"frame_bytes       : {cfg.frame_bytes:,} bytes")
    log.log(f"expected fps      : {cfg.expected_fps:.2f} Hz")
    log.log(f"expected stream   : {cfg.expected_stream_MBps:.2f} MB/s")
    log.log(f"chirp period      : {(cfg.chirp_period_s * 1e6):.1f} us")
    log.log(f"v_max             : {cfg.v_max_mps:.2f} m/s")
    log.log(f"velocity bin      : {cfg.velocity_resolution_mps:.3f} m/s")
    log.log(f"beam cmd file     : {cfg.beam_cmd_file}")
    if cfg.roi_range_m is not None and cfg.roi_range_m >= 0:
        log.log(f"static ROI        : {cfg.roi_range_m:.3f} m +/- {cfg.roi_half_width_m:.3f} m")
    else:
        log.log("static ROI        : disabled; use --roi-range-m after measuring target range")
    log.log(f"static search     : {cfg.static_min_range_m:.2f} m to {cfg.static_max_range_m:.2f} m")
    log.log(f"output dir        : {cfg.output_dir}")
    log.log(f"frame csv         : {cfg.frame_csv}")
    log.log(f"latest json       : {cfg.latest_json}")
    log.log(f"summary json      : {cfg.summary_json}")
    log.log("Run order: receiver -> DCA1000 ARM/StartRecord -> StartFrame -> Lua poller -> angle writer")


def receive_frames(cfg: TestConfig, log: TeeLogger):
    range_axis, velocity_axis = make_axes(cfg)
    frames, raw_frames = [], []
    stats = {
        "total_packets": 0, "packet_gaps": 0, "inserted_missing_bytes": 0, "skipped_overlap_bytes": 0,
        "raw_payload_bytes": 0, "invalid_frames": 0, "valid_frames": 0, "total_frames": 0,
        "first_seq": None, "last_seq": None, "first_byte_count": None, "last_byte_count": None,
        "first_packet_time": None, "last_packet_time": None, "first_frame_time": None, "last_frame_time": None,
        "socket_rcvbuf_actual": None,
    }
    stream_buffer, invalid_mask = bytearray(), bytearray()
    expected_byte_count: Optional[int] = None
    frame_id = 0
    start_time = time.perf_counter()
    last_report_time = start_time

    csv_path = Path(cfg.frame_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FRAME_CSV_FIELDS)
    writer.writeheader(); csv_file.flush()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, cfg.socket_rcvbuf)
        sock.bind((cfg.pc_ip, cfg.data_port))
        sock.settimeout(0.2)
        stats["socket_rcvbuf_actual"] = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        log.log(f"[RX] actual SO_RCVBUF={stats['socket_rcvbuf_actual']}")
        log.log("[RX] listening for DCA1000 packets...")

        while True:
            now = time.perf_counter()
            if (not cfg.infinite) and (now - start_time > cfg.timeout_s) and len(frames) < cfg.frames:
                log.log(f"[RX] timeout after {cfg.timeout_s:.1f}s")
                break
            try:
                data, addr = sock.recvfrom(cfg.recvfrom_bytes)
            except socket.timeout:
                if cfg.infinite and (now - last_report_time >= cfg.report_every_s):
                    elapsed = max(now - start_time, 1e-9)
                    status = {
                        "type": "status", "timestamp_perf": now, "elapsed_s": elapsed,
                        "valid_frames": stats["valid_frames"], "total_frames": stats["total_frames"],
                        "fps": stats["valid_frames"] / elapsed, "expected_fps": cfg.expected_fps,
                        "MBps": (stats["raw_payload_bytes"] / 1e6) / elapsed,
                        "total_packets": stats["total_packets"], "packet_gaps": stats["packet_gaps"],
                        "invalid_frames": stats["invalid_frames"], "missing_bytes": stats["inserted_missing_bytes"],
                        "beam_cmd": read_beam_cmd(cfg.beam_cmd_file),
                    }
                    log.log(f"[RX] status frames={status['valid_frames']} fps={status['fps']:.1f}/{cfg.expected_fps:.1f} MBps={status['MBps']:.2f} beam={status['beam_cmd']}")
                    append_jsonl(cfg.status_jsonl, status); write_json_atomic(cfg.latest_json, status)
                    last_report_time = now
                continue

            src_ip, _ = addr
            if cfg.dca_ip and src_ip != cfg.dca_ip:
                continue
            parsed = parse_dca_packet(data, cfg.dca_header_bytes)
            if parsed is None:
                continue
            seq, byte_count, raw = parsed
            packet_time = time.perf_counter()
            if stats["first_seq"] is None:
                stats["first_seq"] = seq; stats["first_byte_count"] = byte_count; stats["first_packet_time"] = packet_time
                log.log(f"[RX] first packet: seq={seq}, byte_count={byte_count}, payload={len(raw)} bytes")
            if stats["last_seq"] is not None and seq != stats["last_seq"] + 1:
                gap = seq - stats["last_seq"] - 1
                if gap > 0:
                    stats["packet_gaps"] += gap
                    log.log(f"[WARN] packet sequence gap: previous={stats['last_seq']}, current={seq}, missing={gap}")
            stats["last_seq"] = seq; stats["last_byte_count"] = byte_count; stats["last_packet_time"] = packet_time
            stats["total_packets"] += 1; stats["raw_payload_bytes"] += len(raw)

            if expected_byte_count is None:
                offset = byte_count % cfg.frame_bytes
                if offset != 0:
                    skip = cfg.frame_bytes - offset
                    if skip >= len(raw):
                        expected_byte_count = byte_count + len(raw)
                        continue
                    raw = raw[skip:]; byte_count += skip
                    log.log(f"[RX] first packet was mid-frame; skipped {skip} bytes to align")
                expected_byte_count = byte_count

            if byte_count > expected_byte_count:
                missing = byte_count - expected_byte_count
                stream_buffer.extend(b"\x00" * missing); invalid_mask.extend(b"\x01" * missing)
                stats["inserted_missing_bytes"] += missing
                log.log(f"[WARN] byte_count gap: inserted {missing} missing bytes")
                expected_byte_count = byte_count
            elif byte_count < expected_byte_count:
                overlap = expected_byte_count - byte_count
                if overlap >= len(raw):
                    stats["skipped_overlap_bytes"] += len(raw)
                    continue
                stats["skipped_overlap_bytes"] += overlap
                raw = raw[overlap:]; byte_count = expected_byte_count

            stream_buffer.extend(raw); invalid_mask.extend(b"\x00" * len(raw))
            expected_byte_count = byte_count + len(raw)

            while len(stream_buffer) >= cfg.frame_bytes:
                frame_bytes = bytes(stream_buffer[:cfg.frame_bytes])
                invalid_bytes = int(sum(invalid_mask[:cfg.frame_bytes]))
                del stream_buffer[:cfg.frame_bytes]; del invalid_mask[:cfg.frame_bytes]
                t = time.perf_counter(); elapsed = t - start_time
                stats["total_frames"] += 1
                if stats["first_frame_time"] is None: stats["first_frame_time"] = t
                stats["last_frame_time"] = t
                beam_cmd = read_beam_cmd(cfg.beam_cmd_file)
                record = {
                    "frame_id": frame_id, "timestamp_perf": t, "elapsed_s": elapsed,
                    "valid": invalid_bytes == 0, "invalid_bytes": invalid_bytes, "reason": "",
                    **parse_beam_cmd(beam_cmd),
                    "packet_gaps_total": stats["packet_gaps"],
                    "missing_bytes_total": stats["inserted_missing_bytes"],
                    "invalid_frames_total": stats["invalid_frames"],
                }
                if invalid_bytes > 0:
                    stats["invalid_frames"] += 1; record["reason"] = f"missing_or_inserted_bytes={invalid_bytes}"; record["valid"] = False
                    record["invalid_frames_total"] = stats["invalid_frames"]
                    log.log(f"[FRAME {frame_id:04d}] INVALID invalid_bytes={invalid_bytes} beam={beam_cmd}")
                else:
                    try:
                        analysis = analyze_frame(frame_bytes, cfg, range_axis, velocity_axis)
                        record.update(analysis); stats["valid_frames"] += 1
                        if len(raw_frames) < cfg.keep_raw_frames: raw_frames.append(frame_bytes)
                        roi_txt = ""
                        if record.get("static_roi_power_db") is not None:
                            roi_txt = f" ROI={record['static_roi_power_db']:.1f}dB ROI_SNR~={record['static_roi_snr_like_db']:.1f}dB"
                        log.log(
                            f"[FRAME {frame_id:04d}] OK beam={beam_cmd or '-'} "
                            f"raw_std={record['raw_std']:.1f} RD_R={record['rd_peak_range_m']:.2f}m "
                            f"RD_v={record['rd_peak_velocity_mps']:+.2f}m/s RD={record['rd_peak_power_db']:.1f}dB "
                            f"static_R={record['static_peak_range_m']:.2f}m static={record['static_peak_power_db']:.1f}dB{roi_txt}"
                        )
                    except Exception as e:
                        stats["invalid_frames"] += 1; record["valid"] = False; record["reason"] = f"analysis_error: {e}"
                        record["invalid_frames_total"] = stats["invalid_frames"]
                        log.log(f"[FRAME {frame_id:04d}] INVALID analysis_error={e}")
                frames.append(record); writer.writerow({k: record.get(k, "") for k in FRAME_CSV_FIELDS}); csv_file.flush()
                write_json_atomic(cfg.latest_json, {"type": "frame", **record})
                frame_id += 1
                if (not cfg.infinite) and len(frames) >= cfg.frames:
                    return frames, raw_frames, stats

            now = time.perf_counter()
            if cfg.infinite and (now - last_report_time >= cfg.report_every_s):
                elapsed = max(now - start_time, 1e-9)
                status = {
                    "type": "status", "timestamp_perf": now, "elapsed_s": elapsed,
                    "valid_frames": stats["valid_frames"], "total_frames": stats["total_frames"],
                    "fps": stats["valid_frames"] / elapsed, "expected_fps": cfg.expected_fps,
                    "MBps": (stats["raw_payload_bytes"] / 1e6) / elapsed,
                    "total_packets": stats["total_packets"], "packet_gaps": stats["packet_gaps"],
                    "invalid_frames": stats["invalid_frames"], "missing_bytes": stats["inserted_missing_bytes"],
                    "beam_cmd": read_beam_cmd(cfg.beam_cmd_file),
                }
                log.log(f"[RX] running frames={status['valid_frames']} fps={status['fps']:.1f}/{cfg.expected_fps:.1f} MBps={status['MBps']:.2f} gaps={status['packet_gaps']} bad={status['invalid_frames']} beam={status['beam_cmd']}")
                append_jsonl(cfg.status_jsonl, status); write_json_atomic(cfg.latest_json, status)
                last_report_time = now
    except KeyboardInterrupt:
        log.log("[RX] Ctrl+C")
    finally:
        try: sock.close()
        except OSError: pass
        try: csv_file.close()
        except Exception: pass
    return frames, raw_frames, stats


def summarize(cfg: TestConfig, frames: list[dict], stats: dict):
    valid = [f for f in frames if f.get("valid")]
    summary = {
        "config": dataclasses.asdict(cfg), "stats": stats,
        "assembled_frames": len(frames), "valid_frames": len(valid), "invalid_frames": len(frames) - len(valid),
        "frames_csv": cfg.frame_csv, "status_jsonl": cfg.status_jsonl, "latest_json": cfg.latest_json,
        "pass": False, "checks": {},
    }
    checks = summary["checks"]
    checks["got_any_packets"] = stats["total_packets"] > 0
    checks["has_valid_frames"] = len(valid) > 0
    checks["no_overlap_bytes"] = stats["skipped_overlap_bytes"] == 0
    if valid:
        raw_std = np.array([f.get("raw_std") for f in valid if f.get("raw_std") is not None], dtype=float)
        zero_ratio = np.array([f.get("zero_ratio") for f in valid if f.get("zero_ratio") is not None], dtype=float)
        sat_ratio = np.array([f.get("saturation_ratio") for f in valid if f.get("saturation_ratio") is not None], dtype=float)
        rx_span = np.array([f.get("rx_rms_db_span") for f in valid if f.get("rx_rms_db_span") is not None], dtype=float)
        checks["raw_std_nonzero"] = bool(raw_std.size and np.median(raw_std) > 1.0)
        checks["zero_ratio_reasonable"] = bool(zero_ratio.size and np.median(zero_ratio) < 0.50)
        checks["saturation_low"] = bool(sat_ratio.size and np.median(sat_ratio) < 1e-3)
        checks["rx_channels_present"] = bool(rx_span.size and np.median(rx_span) < 40.0)
        summary["median_raw_std"] = float(np.median(raw_std)) if raw_std.size else None
        summary["median_zero_ratio"] = float(np.median(zero_ratio)) if zero_ratio.size else None
        summary["median_saturation_ratio"] = float(np.median(sat_ratio)) if sat_ratio.size else None
        summary["median_rx_rms_db_span"] = float(np.median(rx_span)) if rx_span.size else None
    else:
        checks["raw_std_nonzero"] = checks["zero_ratio_reasonable"] = checks["saturation_low"] = checks["rx_channels_present"] = False
    required = ["got_any_packets", "has_valid_frames", "raw_std_nonzero", "zero_ratio_reasonable", "saturation_low", "rx_channels_present"]
    summary["pass"] = all(bool(checks[k]) for k in required)
    return summary, summary["pass"]


def main() -> int:
    p = argparse.ArgumentParser(description="RadarBox 3Tx simultaneous DCA1000 receive file logger with static ROI metrics")
    p.add_argument("--pc-ip", default="192.168.33.30")
    p.add_argument("--dca-ip", default="192.168.33.180")
    p.add_argument("--data-port", type=int, default=4098)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--infinite", action="store_true")
    p.add_argument("--timeout-s", type=float, default=10.0)
    p.add_argument("--report-every-s", type=float, default=1.0)
    p.add_argument("--loops", type=int, default=64)
    p.add_argument("--tx-groups", type=int, default=1)
    p.add_argument("--rx", type=int, default=4)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--frame-period-ms", type=float, default=20.0)
    p.add_argument("--idle-us", type=float, default=20.0)
    p.add_argument("--ramp-end-us", type=float, default=60.0)
    p.add_argument("--sample-rate-ksps", type=float, default=10000.0)
    p.add_argument("--slope-mhz-us", type=float, default=29.982)
    p.add_argument("--start-freq-ghz", type=float, default=77.0)
    p.add_argument("--socket-rcvbuf", type=int, default=64 * 1024 * 1024)
    p.add_argument("--recvfrom-bytes", type=int, default=9000)
    p.add_argument("--output-dir", default="")
    p.add_argument("--log-file", default="")
    p.add_argument("--frame-csv", default="")
    p.add_argument("--status-jsonl", default="")
    p.add_argument("--latest-json", default="")
    p.add_argument("--summary-json", default="")
    p.add_argument("--save-npz", default=None)
    p.add_argument("--keep-raw-frames", type=int, default=0)
    p.add_argument("--beam-cmd-file", default=r"C:\temp\radarbox_beam_cmd.txt")
    p.add_argument("--roi-range-m", type=float, default=-1.0, help="fixed target range for static beam verification; use -1 to disable")
    p.add_argument("--roi-half-width-m", type=float, default=0.20)
    p.add_argument("--static-min-range-m", type=float, default=0.30)
    p.add_argument("--static-max-range-m", type=float, default=8.00)
    p.add_argument("--no-console", action="store_true")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    cfg = resolve_output_paths(TestConfig(**vars(args)))
    log = TeeLogger(Path(cfg.log_file), no_console=cfg.no_console)
    try:
        print_config(cfg, log)
        frames, raw_frames, stats = receive_frames(cfg, log)
        summary, passed = summarize(cfg, frames, stats)
        Path(cfg.summary_json).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        if cfg.save_npz:
            if raw_frames:
                arr = np.stack([np.frombuffer(b, dtype=np.int16).copy() for b in raw_frames], axis=0)
            else:
                arr = np.zeros((0, cfg.expected_words_per_frame), dtype=np.int16)
            np.savez_compressed(cfg.save_npz, raw_frames_int16=arr, summary_json=json.dumps(summary, default=str))
        log.log("=== Summary ===")
        log.log(f"packets={stats['total_packets']} gaps={stats['packet_gaps']} missing_bytes={stats['inserted_missing_bytes']}")
        log.log(f"assembled={len(frames)} valid={summary['valid_frames']} invalid={summary['invalid_frames']}")
        log.log(f"frames_csv={cfg.frame_csv}")
        log.log(f"summary_json={cfg.summary_json}")
        for k, v in summary["checks"].items():
            log.log(f"check {k}: {'PASS' if v else 'FAIL'}")
        log.log(f"RESULT: {'PASS' if passed else 'FAIL'}")
        return 2 if (cfg.strict and not passed) else 0
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
