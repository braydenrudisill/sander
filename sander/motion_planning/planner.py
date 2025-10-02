import ray
import torch
from typing import cast

from sander.motion_planning.utils import JointState

@ray.remote
def plan_motion(waypoints: list[JointState]) -> list[JointState]:
    return linear_interpolation(waypoints, 0.01)

def linear_interpolation(waypoints: list[JointState], max_step: float) -> list[JointState]:
    """Returns an interpolated collision free trajectory between the waypoints."""
    if len(waypoints) < 2:
        return waypoints

    assert max_step > 0.0

    trajectory: list[JointState] = []

    # Iterate over consecutive waypoint pairs and linearly interpolate in joint space
    for segment_index in range(len(waypoints) - 1):
        start_state = torch.tensor(waypoints[segment_index], dtype=torch.float32)
        end_state = torch.tensor(waypoints[segment_index + 1], dtype=torch.float32)

        delta = end_state - start_state
        segment_length = torch.norm(delta, p=2).item()

        if segment_length == 0.0:
            num_steps = 1
        else:
            num_steps = int(torch.ceil(torch.tensor(segment_length / max_step)).item())
            num_steps = max(1, num_steps)

        for step_index in range(num_steps + 1):
            include_point = segment_index == 0 or step_index > 0
            if not include_point:
                continue

            ratio = step_index / num_steps
            interpolated = start_state + ratio * delta
            trajectory.append(cast(JointState, tuple(float(v) for v in interpolated.tolist())))

    return trajectory
