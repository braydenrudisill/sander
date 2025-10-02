"""
Modified from yourdfpy's vis.py to allow for joint state streaming over a socket.
"""

import sys
import time
import logging
import argparse
import json
import threading
import socketserver
import numpy as np
from functools import partial
from typing import Any, Optional, Dict, Tuple, List, cast

from yourdfpy import __version__
from yourdfpy import URDF

__author__ = "Clemens Eppner"
__copyright__ = "Clemens Eppner"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Visualize a URDF model.")
    parser.add_argument(
        "--version",
        action="version",
        version="yourdfpy {ver}".format(ver=__version__),
    )
    parser.add_argument(
        "--input",
        default="descriptions/annin_ar4/ar.urdf",
        help="URDF file name.",
    )
    parser.add_argument(
        "-c",
        "--configuration",
        nargs="+",
        type=float,
        help="Configuration of the visualized URDF model.",
    )
    parser.add_argument(
        "--collision",
        action="store_true",
        help="Use collision geometry for the visualized URDF model.",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate model by interpolating through all actuated joint limits.",
    )
    parser.add_argument(
        "--joint-state-port",
        type=int,
        default=8765,
        help="Listen for newline-delimited JSON joint states on this TCP port.",
    )
    parser.add_argument(
        "--joint-state-host",
        type=str,
        default="127.0.0.1",
        help="Host interface for TCP joint state listener (default: 127.0.0.1).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging.

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


class JointStateStore:
    """Thread-safe store for the latest joint configuration."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_cfg = None
        self._updated_at = 0.0

    def set_cfg(self, cfg):
        with self._lock:
            self._latest_cfg = cfg
            self._updated_at = time.time()

    def get_latest_cfg(self):
        with self._lock:
            return self._latest_cfg


def _parse_joint_state_message(payload, urdf_model):
    """Parse JSON payload into configuration dict mapping joint->position.

    Supported payloads:
      - {"name": [...], "position": [...]} (ROS-like)
      - {"name": [...], "positions": [...]} (alias)
      - {"joint_a": 0.1, "joint_b": -0.2} (direct map)
    Extra joints are ignored; only actuated joints are used.
    Returns dict or None if invalid.
    """
    if not isinstance(payload, dict):
        return None

    try:
        actuated = set(urdf_model.actuated_joint_names)
    except Exception:
        actuated = None

    # Direct mapping
    direct = {}
    for k, v in payload.items():
        if isinstance(k, str) and k not in ("name", "position", "positions"):
            try:
                direct[k] = float(v)
            except Exception:
                continue
    if direct:
        if actuated is not None:
            direct = {k: v for k, v in direct.items() if k in actuated}
        return direct if direct else None

    # names + positions
    names = payload.get("name")
    positions = payload.get("position", payload.get("positions"))
    if isinstance(names, (list, tuple)) and isinstance(positions, (list, tuple)):
        if len(names) != len(positions):
            return None
        cfg = {}
        for n, p in zip(names, positions):
            if not isinstance(n, str):
                continue
            try:
                cfg[n] = float(p)
            except Exception:
                continue
        if actuated is not None:
            cfg = {k: v for k, v in cfg.items() if k in actuated}
        return cfg if cfg else None

    return None


class JointStateTCPHandler(socketserver.StreamRequestHandler):
    """Reads newline-delimited JSON; updates server.store with parsed cfg."""

    def handle(self):
        try:
            peer = self.request.getpeername()
        except Exception:
            peer = None
        _logger.info("JointState client connected: %s", peer)
        while True:
            line = self.rfile.readline()
            if not line:
                break
            try:
                payload = json.loads(line.decode("utf-8").strip())
            except Exception as e:
                _logger.debug("Invalid JSON from %s: %s", peer, e)
                continue
            server = cast("JointStateTCPServer", self.server)
            cfg = _parse_joint_state_message(payload, server.urdf_model)
            if cfg:
                server.store.set_cfg(cfg)
        _logger.info("JointState client disconnected: %s", peer)


class JointStateTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, server_address: Tuple[str, int], urdf_model: Any, store: JointStateStore):
        super().__init__(server_address, JointStateTCPHandler)
        self.urdf_model = urdf_model
        self.store = store


def start_joint_state_tcp_server(host, port, urdf_model, store):
    server = JointStateTCPServer((host, port), urdf_model=urdf_model, store=store)
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever,
        name=f"JointStateTCPServer@{host}:{port}",
        daemon=True,
    )
    thread.start()
    _logger.info("JointState TCP listener started on %s:%s", host, port)
    return server, thread


def generate_joint_limit_trajectory(urdf_model, loop_time):
    """Generate a trajectory for all actuated joints that interpolates between joint limits.
    For continuous joint interpolate between [0, 2 * pi].

    Args:
        urdf_model (yourdfpy.URDF): _description_
        loop_time (float): Time in seconds to loop through the trajectory.

    Returns:
        dict: A dictionary over all actuated joints with list of configuration values.
    """
    trajectory_via_points = {}
    for joint_name in urdf_model.actuated_joint_names:
        if urdf_model.joint_map[joint_name].type.lower() == "continuous":
            via_point_0 = 0.0
            via_point_2 = 2.0 * np.pi
            via_point_1 = (via_point_2 - via_point_0) / 2.0
        else:
            limit_lower = (
                urdf_model.joint_map[joint_name].limit.lower
                if urdf_model.joint_map[joint_name].limit.lower is not None
                else -np.pi
            )
            limit_upper = (
                urdf_model.joint_map[joint_name].limit.upper
                if urdf_model.joint_map[joint_name].limit.upper is not None
                else +np.pi
            )
            via_point_0 = limit_lower
            via_point_1 = limit_upper
            via_point_2 = limit_lower

        trajectory_via_points[joint_name] = np.array(
            [
                via_point_0,
                via_point_1,
                via_point_2,
            ]
        )
    times = np.linspace(0.0, 1.0, int(loop_time * 100.0))
    bins = np.arange(3) / 2.0

    # Compute alphas for each time
    inds = np.digitize(times, bins, right=True)
    inds[inds == 0] = 1
    alphas = (bins[inds] - times) / (bins[inds] - bins[inds - 1])

    # Create the new interpolated trajectory
    trajectory = {}
    for k in trajectory_via_points:
        trajectory[k] = (
            alphas * trajectory_via_points[k][inds - 1]
            + (1.0 - alphas) * trajectory_via_points[k][inds]
        )

    return trajectory


def viewer_callback(scene, urdf_model, joint_state_store):
    external_cfg = joint_state_store.get_latest_cfg()
    if external_cfg:
        urdf_model.update_cfg(configuration=external_cfg)


def main(args):
    """Wrapper allowing string arguments in a CLI fashion.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    if args.collision:
        urdf_model = URDF.load(
            args.input, build_collision_scene_graph=True, load_collision_meshes=True
        )
    else:
        urdf_model = URDF.load(args.input)

    if args.configuration:
        urdf_model.update_cfg(args.configuration)

    # Optional TCP listener for external joint states
    joint_state_store = None
    print(args.joint_state_port)
    
    joint_state_store = JointStateStore()
    start_joint_state_tcp_server(
        host=args.joint_state_host,
        port=args.joint_state_port,
        urdf_model=urdf_model,
        store=joint_state_store,
    )

    callback = partial(
        viewer_callback,
        urdf_model=urdf_model,
        joint_state_store=joint_state_store,
    )

    urdf_model.show(
        collision_geometry=args.collision,
        callback=callback,
    )


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`.

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()