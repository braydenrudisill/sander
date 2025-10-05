import ray
import pytest

from sander.motion_planning.planner import plan_motion


@pytest.fixture
def simple_waypoints():
    return [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ]

def test_plan(simple_waypoints):
    plan = ray.get(plan_motion.remote(simple_waypoints))
    assert len(plan) > 2
    
    