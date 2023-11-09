import argparse
from pathlib import Path

from translations_parser.parser import TrainingParser


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
    return parser.parse_args()


def main():
    args = get_args()
    with args.input_file.open("r") as f:
        lines = (line.strip() for line in f.readlines())
    parser = TrainingParser(lines)
    parser.parse()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parser.csv_export(args.output_dir)
