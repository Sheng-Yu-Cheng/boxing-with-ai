#!/usr/bin/env python3
r"""
write_beam_angle_sequence.py

Write RadarBox beam phase command file for the mmWave Studio Lua poller.

The Lua poller expects:
    C:\temp\radarbox_beam_cmd.txt

Format:
    tx0Code,tx1Code,tx2Code

Phase quantization:
    1 code = 5.625 deg

Temporary steering model:
    phi_deg = -180 * sin(theta_deg)     assuming d_tx ~= lambda/2
    phases  = [0, phi, 2phi]
    codes   = round(phase / 5.625) mod 64

Examples:
    python write_beam_angle_sequence.py --theta 0
    python write_beam_angle_sequence.py --theta 30
    python write_beam_angle_sequence.py --sequence 0,30,0,-30,0 --hold-s 5
    python write_beam_angle_sequence.py --interactive
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

PHASE_STEP_DEG = 5.625


def deg_to_code(deg: float) -> int:
    return int(round((deg % 360.0) / PHASE_STEP_DEG)) & 0x3F


def theta_to_codes(theta_deg: float) -> tuple[int, int, int, tuple[float, float, float]]:
    phi = -180.0 * math.sin(math.radians(theta_deg))
    tx0_deg = 0.0
    tx1_deg = phi
    tx2_deg = 2.0 * phi
    return (
        deg_to_code(tx0_deg),
        deg_to_code(tx1_deg),
        deg_to_code(tx2_deg),
        (tx0_deg % 360.0, tx1_deg % 360.0, tx2_deg % 360.0),
    )


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="ascii")
    tmp.replace(path)


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_theta(theta_deg: float, cmd_path: Path, log_path: Path) -> None:
    c0, c1, c2, phases = theta_to_codes(theta_deg)
    cmd = f"{c0},{c1},{c2}\n"
    atomic_write(cmd_path, cmd)

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{ts},theta_deg={theta_deg:.2f},"
        f"phase_deg=[{phases[0]:.1f},{phases[1]:.1f},{phases[2]:.1f}],"
        f"codes=[{c0},{c1},{c2}],cmd_path={cmd_path}"
    )
    append_log(log_path, line)
    print(line)


def parse_sequence(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="Write RadarBox beam phase command file")
    p.add_argument("--cmd-path", default=r"C:\temp\radarbox_beam_cmd.txt")
    p.add_argument("--log-path", default=r"C:\temp\radarbox_beam_command_log.csv")

    group = p.add_mutually_exclusive_group()
    group.add_argument("--theta", type=float, help="write one theta command and exit")
    group.add_argument("--sequence", default=None, help="comma-separated theta sequence, e.g. 0,30,0,-30,0")
    group.add_argument("--interactive", action="store_true", help="prompt for theta values interactively")

    p.add_argument("--hold-s", type=float, default=5.0, help="hold time between sequence steps")
    args = p.parse_args()

    cmd_path = Path(args.cmd_path)
    log_path = Path(args.log_path)

    if args.theta is not None:
        write_theta(args.theta, cmd_path, log_path)
        return 0

    if args.sequence:
        seq = parse_sequence(args.sequence)
        for i, theta in enumerate(seq):
            write_theta(theta, cmd_path, log_path)
            if i != len(seq) - 1:
                time.sleep(args.hold_s)
        return 0

    if args.interactive:
        print("Interactive beam writer. Enter theta in degrees, or q to quit.")
        while True:
            s = input("theta_deg> ").strip()
            if s.lower() in ("q", "quit", "exit"):
                break
            try:
                theta = float(s)
            except ValueError:
                print("Invalid number.")
                continue
            write_theta(theta, cmd_path, log_path)
        return 0

    write_theta(0.0, cmd_path, log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
