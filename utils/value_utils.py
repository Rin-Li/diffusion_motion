"""
Backwards-compatibility shim.
Import from the canonical modules instead:
  from utils.scenario_utils import create_test_scenario, generate_path
  from utils.viz_utils       import show_multiple_with_collision_colors
                                     visualize_result
                                     visualize_trajectory_gif
"""
from utils.scenario_utils import create_test_scenario, generate_path
from utils.viz_utils import (
    show_multiple_with_collision_colors,
    visualize_result,
    visualize_trajectory_gif,
)

__all__ = [
    "create_test_scenario",
    "generate_path",
    "show_multiple_with_collision_colors",
    "visualize_result",
    "visualize_trajectory_gif",
]
