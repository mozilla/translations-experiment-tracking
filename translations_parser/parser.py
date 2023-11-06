import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Training:
    epoch: str
    up: str
    sen: str
    cost: str
    time: int
    rate: str
    gnorm: str = None


@dataclass
class Validation:
    epoch: int
    up: str
    chrf: str
    ce_mean_words: str
    bleu_detok: str


@dataclass
class TrainingLog:
    """Results from the parsing of a training log file"""

    configuration: dict
    training: List[Training]
    validation: List[Validation]
    # Dict of log lines indexed by their header (e.g. marian, data, memory)
    logs: dict


class TrainingParser:
    def __init__(self, logs_iter):
        # Iterable reading logs lines
        self.logs_iter = logs_iter
        self.output: List[TrainingLog] = None

    def parse(self):
        # TODO: Generate entries from log lines
        self.output = [
            TrainingLog(
                configuration={"test": "a"},
                training=[Training(1, 2, 3, 4, 5, 6)],
                validation=[],
                logs={"head": "val"},
            )
        ]


def main():
    with (Path(__file__).parent.parent / "samples" / "KZPjvTEiSmO--BXYpQCNPQ.txt").open("r") as f:
        lines = (line.strip() for line in f.readlines())
    parser = TrainingParser(lines)
    parser.parse()
    logger.info(parser.output)
