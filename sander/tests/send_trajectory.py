import argparse
import json
import socket
import time
from typing import List

import ray

from sander.motion_planning.planner import plan_motion
from sander.motion_planning.utils import JointState


JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]


def _default_waypoints() -> List[JointState]:
    return [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.5, -0.5, 0.75, 0.0, 0.25, -0.25),
        (1.0, 0.5, 1.25, -0.5, 0.5, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]


def stream_trajectory(host: str, port: int, trajectory: List[JointState], rate_hz: float) -> None:
    period_s = 1.0 / max(rate_hz, 1e-6)
    with socket.create_connection((host, port)) as sock:
        for point in trajectory:
            message = {name: float(val) for name, val in zip(JOINT_NAMES, point)}
            data = json.dumps(message).encode("utf-8") + b"\n"
            sock.sendall(data)
            time.sleep(period_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a simple trajectory and send it over TCP.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Joint state server host.")
    parser.add_argument("--port", type=int, default=8765, help="Joint state server TCP port.")
    parser.add_argument("--rate", type=float, default=200.0, help="Streaming rate in Hz.")
    args = parser.parse_args()

    # Initialize Ray (no-op if already initialized)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    waypoints = _default_waypoints()
    trajectory = ray.get(plan_motion.remote(waypoints))

    stream_trajectory(args.host, args.port, trajectory, args.rate)


if __name__ == "__main__":
    main()


