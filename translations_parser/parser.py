import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Callable

import yaml

from translations_parser.data import TrainingEpoch, TrainingLog, ValidationEpoch
from translations_parser.publishers import Publisher

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
    r"Time[ :]+(?P<time>[\d\.]+)s[ :]+"
    r"(?P<rate>[\d\.]+) words\/s[ :]+"
    r"gNorm[ :]+(?P<gnorm>[\d\.]+)"
)
# Expected version of Marian for a clean parsing
MARIAN_MAJOR, MARIAN_MINOR = 1, 10


class TrainingParser:
    def __init__(
        self,
        logs_iter: Iterable[str],
        publishers: Sequence[Publisher],
        log_filter: Callable = None,
        skip_marian_context: bool = False,
    ):
        # Iterable reading logs lines
        self.logs_iter = logs_iter
        # Function to exclude log lines depending on the headers
        self.log_filter = log_filter
        self._current_index = 0
        self.parsed = False
        self.config = {}
        self.indexed_logs = defaultdict(list)
        # List of TrainingEpoch
        self.training = []
        # List of ValidationEpoch
        self.validation = []
        # Dict mapping (epoch, up) to values parsed on multiple lines
        self._validation_entries = defaultdict(dict)
        # Marian exection data
        self.skip_marian_context = skip_marian_context
        self.version = None
        self.version_hash = None
        self.release_date = None
        self.run_date = None
        self.description = None
        # Data publication after parsing logs
        self.publishers = publishers

    def get_headers(self, line):
        """
        Returns a list of tuples representing the headers of a log line
        and the position of the last index
        """
        matches = list(HEADER_RE.finditer(line))
        if not matches:
            return ((), None)
        return ([tuple(m.group("value").split()) for m in matches], matches[-1].span()[-1])

    def get_timestamp(self, headers):
        """
        Look for a timestamp in header tags.
        Returns the timestamp if found, None otherwise.
        """
        for values in headers:
            if len(values) != 2:
                continue
            base, timestamp = values
            # TC adds a timestamp after the task header
            if base == "task":
                try:
                    return datetime.fromisoformat(timestamp.rstrip("Z"))
                except ValueError:
                    pass
            # Marian timestamp is composed of two values, one for date and one for hour precision
            try:
                return datetime.fromisoformat("T".join(values))
            except ValueError:
                pass

    def parse_training_log(self, headers, text):
        match = TRAINING_RE.match(text)
        if not match:
            return
        values = match.groupdict()
        # Update sen value from 1,234,567 to 1_234_567 that Python interprets
        values["sen"] = values["sen"].replace(",", "_")
        # Transform values to match output types
        values = {k: TrainingEpoch.__annotations__[k](v) for k, v in values.items()}
        training_epoch = TrainingEpoch(**values)
        self.training.append(training_epoch)
        return training_epoch

    def parse_validation_log(self, headers, text):
        if ("valid",) not in headers or not (match := VALIDATION_RE.match(text)):
            return
        epoch, up, key, val = match.groups()
        # Replace items keys to match ValidationEpoch dataclass
        key = key.replace("-", "_")
        # Transform values to match output types
        epoch, up = int(epoch), int(up)
        val = ValidationEpoch.__annotations__[key](val)
        self._validation_entries[(epoch, up)].update({key: val})
        return (epoch, up)

    def _iter_log_entries(self):
        for line in self.logs_iter:
            self._current_index += 1
            headers, position = self.get_headers(line)
            if self.log_filter and not self.log_filter(headers):
                logger.debug(f"Skipping line {self._current_index} : Headers does not match the filter")
                continue
            elif self.run_date is None:
                # Try to fill run date from log headers
                self.run_date = self.get_timestamp(headers)
            text = line[position:]

            # Record logs depending on Marian headers
            if len(headers) >= 2:
                # First is task timestamp, second is marian timestamp
                _, _, *marian_tags = headers
                tag = "_".join(*marian_tags) if marian_tags else "_"
                self.indexed_logs[tag].append(text)

            yield headers, text

    def _parse(self):
        if self.parsed:
            raise Exception("The parser already ran.")
        logs_iter = self._iter_log_entries()

        if self.skip_marian_context:
            headers, text = next(logs_iter)
        else:
            # Consume first lines until we get the Marian header
            headers = []
            while ("marian",) not in headers:
                headers, text = next(logs_iter)

            # Read Marian runtime logs
            _, version, self.version_hash, self.release_date, *_ = text.split()
            self.version = version.rstrip(";")
            major, minor = map(int, version.lstrip("v").split(".")[:2])
            if (major, minor) > (MARIAN_MAJOR, MARIAN_MINOR):
                logger.warning(
                    f"Parsing logs from a newer version of Marian ({major}.{minor} > {MARIAN_MAJOR}.{MARIAN_MINOR})"
                )

            # Read Marian execution description on the next lines
            desc = []
            for headers, text in logs_iter:
                if ("marian",) not in headers:
                    break
                desc.append(text)
            self.description = " ".join(desc)

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
            try:
                try:
                    training = self.parse_training_log(headers, text)
                    if not training:
                        self.parse_validation_log(headers, text)
                except ValueError as e:
                    logger.warning(f"Line {self._current_index} could not be stored: {e}.")
                    headers, text = next(logs_iter)
                finally:
                    headers, text = next(logs_iter)
            except StopIteration:
                break

        # Build validation epochs from matched log entries
        for validation in self.build_validation_epochs():
            self.validation.append(validation)
        self.parsed = True

    def build_validation_epochs(self):
        """
        Build validation entries from complete entries
        as validation logs are displayed on multiple lines
        """
        for (epoch, up), parsed in self._validation_entries.items():
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
            run_date=self.run_date,
            configuration=self.config,
            training=self.training,
            validation=list(self.validation),
            logs=self.indexed_logs,
        )

    def publish(self, publisher):
        logger.info(f"Publishing data using {publisher.__class__.__name__}")
        publisher.publish(self.output)
        publisher.close()

    def run(self):
        """
        Parse the log lines.
        """
        try:
            self._parse()
        except StopIteration:
            # A StopIteration can be raised if some required lines are never found.
            raise ValueError("Logs file ended up unexpectedly")

        count = sum(len(vals) for vals in self.indexed_logs.values())
        logger.info(f"Successfully parsed {count} lines")
        logger.info(f"Found {len(self.training)} training entries")
        logger.info(f"Found {len(self.validation)} validation entries")

        for publisher in self.publishers:
            self.publish(publisher)
