import argparse
from pathlib import Path

from translations_parser.parser import TrainingParser
from translations_parser.publishers import CSVExport, WandB


def get_args():
    parser = argparse.ArgumentParser(description="Extract information from Marian execution on Task Cluster")
    parser.add_argument(
        "--input-file",
        "-i",
        help="Path to the Task Cluster log file.",
        type=Path,
        default=Path(__file__).parent.parent / "samples" / "KZPjvTEiSmO--BXYpQCNPQ.txt",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory to export training and validation data as CSV.",
        type=Path,
        default=Path(__file__).parent.parent / "output",
    )
    parser.add_argument(
        "--wandb-project",
        help="Publish the training run to a Weight & Biases project.",
        default=None,
    )
    parser.add_argument(
        "--wandb-group",
        help="Add the training run to a Weight & Biases group e.g. by language pair or experiment.",
        default=None,
    )
    parser.add_argument(
        "--wandb-run-name",
        help="Use a custom name for the Weight & Biases run.",
        default=None,
    )
    return parser.parse_args()


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.input_file.open("r") as f:
        lines = (line.strip() for line in f.readlines())
    publishers = [CSVExport(output_dir=args.output_dir)]
    if args.wandb_project:
        publishers.append(
            WandB(
                project=args.wandb_project,
                group=args.wandb_group,
                tags=["cli"],
                name=args.wandb_run_name,
                config={
                    "logs_file": args.input_file,
                },
            )
        )
    parser = TrainingParser(lines, publishers=publishers)
    parser.run()
