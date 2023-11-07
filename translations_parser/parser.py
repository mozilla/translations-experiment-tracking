import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

HEADER_RE = re.compile(r"(?<=\[)(?P<value>.+?)\] ")
VALIDATION_RE = re.compile(r"Ep\.[ :]+(?P<ep>\d+)[ :]+Up\.[ :]+(?P<up>\d+)[ :]+(?P<key>[\w-]+)[ :]+(?P<value>[\d\.]+)")
TRAINING_RE = re.compile(
    r"Ep\.[ :]+(?P<epoch>\d+)[ :]+"
    r"Up\.[ :]+(?P<up>\d+)[ :]+"
    r"Sen\.[ :]+(?P<sen>[\d,]+)[ :]+"
    r"Cost[ :]+(?P<cost>[\d.]+)[ :]+"
    r"Time[ :]+(?P<time>[\d\.]+s)[ :]+"
    r"(?P<rate>[\d\.]+ words\/s)[ :]+"
    r"gNorm[ :]+(?P<gnorm>[\d\.]+)"
)
# Expected version of Marian for a clean parsing
MARIAN_MAJOR, MARIAN_MINOR = 1, 10


@dataclass
class TrainingEpoch:
    epoch: str
    up: str
    sen: str
    cost: str
    time: int
    rate: str
    gnorm: str


@dataclass
class ValidationEpoch:
    epoch: str
    up: str
    chrf: str
    ce_mean_words: str
    bleu_detok: str


@dataclass
class TrainingLog:
    """Results from the parsing of a training log file"""

    # Marian information
    info: dict
    # Runtime configuration
    configuration: dict
    training: List[TrainingEpoch]
    validation: List[ValidationEpoch]
    # Dict of log lines indexed by their header (e.g. marian, data, memory)
    logs: dict


class TrainingParser:
    def __init__(self, logs_iter):
        # Iterable reading logs lines
        self.logs_iter = logs_iter
        self.parsed = False
        self.config = {}
        self.indexed_logs = defaultdict(list)
        self.training = []
        # Dict mapping (epoch, up) to values parsed on multiple lines
        self.validation_entries = defaultdict(dict)
        # Marian exection data
        self.version = None
        self.version_hash = None
        self.release_date = None

    def get_headers(self, line):
        """
        Returns a list of tuples representing the headers of a log line
        and the position of the last index
        """
        matches = list(HEADER_RE.finditer(line))
        if not matches:
            return ((), None)
        return ([tuple(m.group("value").split()) for m in matches], matches[-1].span()[-1])

    def check_task_timestamp_header(self, values):
        """
        Check a header value matching ('task', <timestamp>)
        and return the deduced timestamp
        """
        if not values or len(values) != 2:
            return
        base, timestamp = values
        if base != "task":
            return
        return datetime.fromisoformat(timestamp.rstrip("Z"))

    def parse_training_log(self, headers, text):
        match = TRAINING_RE.match(text)
        if not match:
            return
        self.training.append(TrainingEpoch(**match.groupdict()))

    def parse_validation_log(self, headers, text):
        if ("valid",) not in headers:
            return
        match = VALIDATION_RE.match(text)
        if not match:
            return
        epoch, up, key, val = match.groups()
        # Replace items keys to match ValidationEpoch dataclass
        key = key.replace("-", "_")
        self.validation_entries[(epoch, up)].update({key: val})

    def _iter_log_entries(self):
        for index, line in enumerate(self.logs_iter, start=1):
            headers, position = self.get_headers(line)
            timestamp = next((ts for ts in map(self.check_task_timestamp_header, headers) if ts), None)
            if timestamp is None:
                logger.debug(f"Skipping line {index} : Headers does not match [task <timestamp>]")
                continue
            text = line[position:]

            # Record logs depending on Marian headers
            if len(headers) >= 2:
                # First is task timestamp, second is marian timestamp
                _, _, *marian_tags = headers
                tag = "_".join(*marian_tags) if marian_tags else "_default"
                self.indexed_logs[tag].append(text)

            yield headers, text

    def _parse(self):
        if self.parsed:
            raise Exception("The parser already ran.")
        logs_iter = self._iter_log_entries()

        # Consume first lines until we get the Marian header
        headers = []
        while ("marian",) not in headers:
            headers, text = next(logs_iter)

        # Read Marian runtime logs
        _, version, self.version_hash, self.release_date, *_ = text.split()
        self.version = version.rstrip(";")
        major, minor = map(int, version.lstrip("v").split(".")[:2])
        if (major, minor) > (MARIAN_MAJOR, MARIAN_MINOR):
            logger.warning("Parsing logs from a newer version of Marian (> {MARIAN_MAJOR}.{MARIAN_MINOR})")

        # Read Marian execution description on the next lines
        desc = []
        for headers, text in logs_iter:
            if ("marian",) not in headers:
                break
            desc.append(text)

        # Try to parse all following config lines as YAML
        config_yaml = ""
        while ("config",) in headers:
            if "Model is being created" in text:
                headers, text = next(logs_iter)
                break
            config_yaml += f"{text}\n"
            headers, text = next(logs_iter)
        try:
            self.config = yaml.safe_load(config_yaml)
        except Exception as e:
            raise Exception(f"Invalid config section: {e}")

        # Iterate until the end of file to find training or validation logs
        while True:
            if train := self.parse_training_log(headers, text):
                self.training.append(train)
            elif val := self.parse_validation_log(headers, text):
                self.validation.append(val)
            try:
                headers, text = next(logs_iter)
            except StopIteration:
                break

        count = sum(len(vals) for vals in self.indexed_logs.values())
        logger.info(f"Successfully parsed {count} lines")
        logger.info(f"Found {len(self.training)} training entries")
        logger.info(f"Found {len(list(self.validation))} validation entries")
        self.parsed = True

    def parse(self):
        """
        Parse the log lines
        A StopIteration can be raised if some required lines are never found
        """
        try:
            self._parse()
        except StopIteration:
            raise ValueError("Logs file ended up unexpectedly")

    @property
    def validation(self):
        """
        Build validation entries from complete entries
        as validation logs are displayed on multiple lines
        """
        for (epoch, up), parsed in self.validation_entries.items():
            # Ensure required keys have been parsed
            diff = set(("chrf", "ce_mean_words", "bleu_detok")) - set(parsed.keys())
            if diff:
                logger.warning(f"Missing keys for validation entry ep. {epoch} up. {up}: {diff}")
                continue
            yield ValidationEpoch(epoch=epoch, up=up, **parsed)

    @property
    def output(self):
        if not self.parsed:
            raise Exception("Please run the parser before reading the output")
        return TrainingLog(
            info=self.info,
            configuration=self.config,
            training=self.training,
            validation=list(self.validation),
            logs=self.indexed_logs,
        )

    def csv_export(self):
        output = Path(__file__).parent.parent / "output"
        output.mkdir(exist_ok=True)
        # Publish two files, validation.csv and training.csv
        training_output = output / "training.csv"
        if training_output.exists():
            print(f"Output file {training_output} exists, skipping.")
        else:
            with open(training_output, "w") as f:
                writer = csv.DictWriter(f, fieldnames=TrainingEpoch.__annotations__)
                writer.writeheader()
                for entry in self.training:
                    writer.writerow(vars(entry))

        validation_output = output / "validation.csv"
        if validation_output.exists():
            print(f"Output file {validation_output} exists, skipping.")
        else:
            with open(validation_output, "w") as f:
                writer = csv.DictWriter(f, fieldnames=ValidationEpoch.__annotations__)
                writer.writeheader()
                for entry in self.validation:
                    writer.writerow(vars(entry))
