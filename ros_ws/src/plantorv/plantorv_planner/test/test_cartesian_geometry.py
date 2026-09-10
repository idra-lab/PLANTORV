"""
Copyright 2025 Enrico Saccon

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__maintainers__ = ["Enrico Saccon", "Tommaso Faraci"]

"""The maths behind a streamed straight line, tested without a robot.

Nothing here needs ROS running. The interpolation and the timing are the parts
of the Cartesian back end that can be wrong quietly -- a profile that exceeds
the speed it was given produces a move that works in the simulator and is too
fast on the real arm -- so they are checked numerically, against the limits
they are supposed to respect rather than against remembered numbers.
"""

import math

from plantorv_planner.cartesian_client import (
    _angle_between,
    _distance,
    _duration,
    _eased,
    _interpolate,
    _samples,
    _slerp,
)
from plantorv_planner.geometry import pose_above, top_down_quaternion


def _peak_rates(distance, duration, steps=20000):
    """Peak speed and acceleration of the eased profile, by differences."""
    step = duration / steps
    positions = [distance * _eased(index * step / duration) for index in range(steps + 1)]
    speeds = [(b - a) / step for a, b in zip(positions, positions[1:])]
    accels = [(b - a) / step for a, b in zip(speeds, speeds[1:])]
    return max(abs(v) for v in speeds), max(abs(a) for a in accels)


class TestEasing:
    def test_starts_and_ends_at_the_ends(self):
        assert _eased(0.0) == 0.0
        assert math.isclose(_eased(1.0), 1.0)

    def test_is_monotone(self):
        values = [_eased(index / 100.0) for index in range(101)]
        assert all(b >= a for a, b in zip(values, values[1:]))

    def test_is_symmetric_about_the_middle(self):
        for fraction in (0.1, 0.25, 0.4):
            assert math.isclose(_eased(fraction) + _eased(1.0 - fraction), 1.0)

    def test_starts_and_ends_at_rest(self):
        # Which is the whole reason for using it: a target that moves off at
        # speed asks the controller for a step change and gets a lurch.
        assert _eased(0.001) < 0.001
        assert 1.0 - _eased(0.999) < 0.001


class TestDuration:
    def test_respects_the_speed_limit(self):
        # Long and gentle: speed is what binds.
        distance, speed, accel = 0.40, 0.10, 10.0
        duration = _duration(distance, 0.0, speed, accel, rot_speed=1.0)
        peak_speed, _ = _peak_rates(distance, duration)
        assert peak_speed <= speed * 1.001
        # And is not needlessly slower than it has to be.
        assert peak_speed > speed * 0.99

    def test_respects_the_acceleration_limit(self):
        # Short and sharp: acceleration is what binds.
        distance, speed, accel = 0.01, 10.0, 0.25
        duration = _duration(distance, 0.0, speed, accel, rot_speed=1.0)
        _, peak_accel = _peak_rates(distance, duration)
        assert peak_accel <= accel * 1.01

    def test_a_rotation_alone_still_takes_time(self):
        duration = _duration(0.0, math.pi / 2.0, speed=0.1, accel=0.25, rot_speed=0.8)
        assert duration > 0.0

    def test_nothing_to_do_takes_no_time(self):
        assert _duration(0.0, 0.0, speed=0.1, accel=0.25, rot_speed=0.8) == 0.0

    def test_a_slower_limit_takes_longer(self):
        fast = _duration(0.3, 0.0, speed=0.20, accel=10.0, rot_speed=1.0)
        slow = _duration(0.3, 0.0, speed=0.05, accel=10.0, rot_speed=1.0)
        assert slow > fast


class TestOrientation:
    def test_no_angle_between_an_orientation_and_itself(self):
        q = top_down_quaternion(0.3)
        assert math.isclose(_angle_between(q, q), 0.0, abs_tol=1e-9)

    def test_a_quarter_turn_of_yaw_is_a_quarter_turn(self):
        angle = _angle_between(top_down_quaternion(0.0), top_down_quaternion(math.pi / 2.0))
        assert math.isclose(angle, math.pi / 2.0, abs_tol=1e-9)

    def test_slerp_hits_both_ends(self):
        a, b = top_down_quaternion(0.0), top_down_quaternion(0.9)
        assert math.isclose(_angle_between(_slerp(a, b, 0.0), a), 0.0, abs_tol=1e-9)
        assert math.isclose(_angle_between(_slerp(a, b, 1.0), b), 0.0, abs_tol=1e-9)

    def test_slerp_halfway_is_halfway(self):
        a, b = top_down_quaternion(0.0), top_down_quaternion(1.0)
        middle = _slerp(a, b, 0.5)
        assert math.isclose(_angle_between(a, middle), _angle_between(middle, b), abs_tol=1e-9)

    def test_slerp_stays_a_unit_quaternion(self):
        a, b = top_down_quaternion(0.0), top_down_quaternion(1.2)
        for index in range(11):
            q = _slerp(a, b, index / 10.0)
            assert math.isclose(math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2), 1.0, abs_tol=1e-9)

    def test_slerp_takes_the_short_way_round(self):
        # A quaternion and its negation are the same orientation, so
        # interpolating towards the far one must not wind the wrist the long
        # way. The folded yaws in geometry.py mean this comes up.
        a = top_down_quaternion(0.0)
        b = top_down_quaternion(0.4)
        far = type(b)(x=-b.x, y=-b.y, z=-b.z, w=-b.w)
        near_path = _angle_between(a, _slerp(a, b, 0.5))
        far_path = _angle_between(a, _slerp(a, far, 0.5))
        assert math.isclose(near_path, far_path, abs_tol=1e-9)


class TestInterpolation:
    def test_halfway_along_a_line_is_the_midpoint(self):
        start = pose_above(0.0, 0.0, 1.30)
        goal = pose_above(0.20, 0.10, 1.30)
        middle = _interpolate(start, goal, 0.5)
        assert math.isclose(middle.position.x, 0.10)
        assert math.isclose(middle.position.y, 0.05)
        assert math.isclose(middle.position.z, 1.30)

    def test_a_traverse_between_two_points_on_the_plane_stays_on_it(self):
        # The invariant the whole action library rests on, now that nothing
        # checks collisions: equal heights at both ends, so no sample dips.
        start = pose_above(-0.20, 0.15, 1.32)
        goal = pose_above(0.25, -0.10, 1.32)
        for sample in _samples(start, goal, step=0.005):
            assert math.isclose(sample.position.z, 1.32, abs_tol=1e-12)


class TestSamples:
    def test_includes_both_ends(self):
        start = pose_above(0.0, 0.0, 1.10)
        goal = pose_above(0.30, 0.0, 1.10)
        samples = _samples(start, goal, step=0.005)
        assert math.isclose(_distance(samples[0].position, start.position), 0.0, abs_tol=1e-12)
        assert math.isclose(_distance(samples[-1].position, goal.position), 0.0, abs_tol=1e-12)

    def test_never_spaced_wider_than_the_step(self):
        start = pose_above(0.0, 0.0, 1.10)
        goal = pose_above(0.137, 0.041, 1.29)
        samples = _samples(start, goal, step=0.005)
        for a, b in zip(samples, samples[1:]):
            assert _distance(a.position, b.position) <= 0.005 + 1e-12

    def test_a_zero_length_line_is_still_one_point_at_each_end(self):
        here = pose_above(0.1, 0.1, 1.2)
        samples = _samples(here, here, step=0.005)
        assert len(samples) == 2
