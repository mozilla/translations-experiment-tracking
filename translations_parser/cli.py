import argparse
import csv
from pathlib import Path

from translations_parser.parser import Training, TrainingParser, Validation


def get_args():
    parser = argparse.ArgumentParser(description="Extract information from Marian execution on Task Cluster")
    parser.add_argument(
        "--input-file",
        "-i",
        help="Path to the Task Cluster log file.",
        type=Path,
        default=Path(__file__).parent.parent / "samples" / "KZPjvTEiSmO--BXYpQCNPQ.txt",
    )
    return parser.parse_args()


def main():
    args = get_args()
    with args.input_file.open("r") as f:
        lines = (line.strip() for line in f.readlines())
    parser = TrainingParser(lines)
    parser.parse()

    output = Path(__file__).parent.parent / "output"
    output.mkdir(exist_ok=True)
    # Publish two files, validation.csv and training.csv
    training_output = output / "training.csv"
    if training_output.exists():
        print(f"Output file {training_output} exists, skipping.")
    else:
        with open(training_output, "w") as f:
            writer = csv.DictWriter(f, fieldnames=Training.__annotations__)
            writer.writeheader()
            for entry in parser.output.training:
                writer.writerow(vars(entry))

    validation_output = output / "validation.csv"
    if validation_output.exists():
        print(f"Output file {validation_output} exists, skipping.")
    else:
        with open(validation_output, "w") as f:
            writer = csv.DictWriter(f, fieldnames=Validation.__annotations__)
            writer.writeheader()
            for entry in parser.output.validation:
                writer.writerow(vars(entry))
