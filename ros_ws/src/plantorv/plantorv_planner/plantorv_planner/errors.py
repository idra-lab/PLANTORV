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

"""The one error the planner raises.

It used to live in moveit_client, which was fine while MoveIt was the only
back end. Now that a motion can fail inside the Cartesian client instead, the
error belongs to neither of them; both raise it, and actions.py lets it out.
"""


class PlanningError(RuntimeError):
    """A motion could not be produced or executed. The message says why."""
