import ray
import pytest

from sander.motion_planning.planner import plan_motion


@pytest.fixture
def simple_waypoints():
    return [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ]

def test_plan_simple(simple_waypoints):
    plan = ray.get(plan_motion.remote(simple_waypoints, 0.1))
    assert len(plan) > 2
    
def test_plan_huge_step(simple_waypoints):
    plan = ray.get(plan_motion.remote(simple_waypoints, 1000.0))
    assert len(plan) == 2

    