"""Run the whole pipeline: segmentation, annotation and depth, in that order.

Kept at the repository root because the ROS nodes and the cluster jobs call it by
path, and the ROS scene service finds the repository by this file. The pipeline
itself lives in :mod:`pipeline.run`; ``python3 samgpt.py --help`` lists its options.
"""

from pipeline.run import run_cli

if __name__ == "__main__":
    run_cli()
