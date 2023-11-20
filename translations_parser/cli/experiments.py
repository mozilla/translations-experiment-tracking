import argparse
import logging
import os
from itertools import groupby
from pathlib import Path

from translations_parser.parser import TrainingParser
from translations_parser.publishers import WandB

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Publish multiple experiments to Weight & Biases")
    parser.add_argument(
        "--directory",
        "-d",
        help="Path to the experiments directory.",
        type=Path,
        default=Path(Path(os.getcwd())),
    )
    return parser.parse_args()


def parse_experiment(logs_file, project, group, name, tags=[]):
    with logs_file.open("r") as f:
        lines = (line.strip() for line in f.readlines())
    parser = TrainingParser(
        lines,
        publishers=[
            WandB(
                project=project,
                name=name,
                group=group,
                tags=tags,
                config={"logs_file": logs_file},
            )
        ],
    )
    parser.run()


def main():
    args = get_args()
    directory = args.directory
    file_groups = {path: list(files) for path, files in groupby(directory.glob("**/*.log"), lambda path: path.parent)}
    total_count = len(file_groups)
    # Exclude groups missing valid or train log files
    file_groups = {
        path: files
        for path, files in file_groups.items()
        if ["train.log", "valid.log"] == sorted(f.name for f in files)
    }
    logger.info(f"Reading {len(file_groups)} training data (over {total_count} folders)")
    prefix = os.path.commonprefix([path.parts for path in file_groups])
    for path, files in file_groups.items():
        parents = path.parts[len(prefix) :]
        if len(parents) < 3:
            logger.warning(f"Skipping folder {path}: Unexpected folder structure")
            continue
        project, group, *name = parents
        name = "_".join(name)
        # Parse logs
        for file in files:
            try:
                parse_experiment(file, project, group, name)
            except Exception as e:
                logger.error(f"An exception occured parsing {file}: {e}")
