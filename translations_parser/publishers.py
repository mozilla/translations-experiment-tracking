import csv
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import wandb

from translations_parser.data import TrainingEpoch, TrainingLog, ValidationEpoch

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class Publisher(ABC):
    @abstractmethod
    def publish(log: TrainingLog):
        ...

    def close(self):
        ...


class CSVExport(Publisher):
    def __init__(self, output_dir):
        assert output_dir.is_dir(), "Output must be a valid directory"
        self.output_dir = output_dir

    def write_data(output, entries, dataclass):
        if not entries:
            logger.warning(f"No {dataclass.__name__} entry, skipping.")
        with open(output, "w") as f:
            writer = csv.DictWriter(f, fieldnames=dataclass.__annotations__)
            writer.writeheader()
            for entry in entries:
                writer.writerow(vars(entry))

    def publish(self, training_log):
        training_output = self.output_dir / "training.csv"
        if training_output.exists():
            logger.warning(f"Training output file {training_output} exists, skipping.")
        else:
            self.write_data(training_output, training_log.training, TrainingEpoch)

        validation_output = self.output_dir / "validation.csv"
        if validation_output.exists():
            logger.warning(f"Validation output file {validation_output} exists, skipping.")
        else:
            self.write_data(validation_output, training_log.validation, ValidationEpoch)


class WanDB(Publisher):
    def __init__(self, project, **extra_kwargs):
        self.project = project
        self.extra_kwargs = extra_kwargs

    def publish(self, training_log):
        # Weight & Biases requires to publish data ordered by epoch
        data = sorted([*training_log.training, *training_log.validation], key=lambda epoch: epoch.up)
        if not data:
            logger.warning("No data to push, skipping.")
            return

        config = training_log.configuration
        config.update(self.extra_kwargs.pop("config", {}))

        # Start a W&B run and publish data for training and validation jobs
        self.wandb = wandb.init(
            project=self.project,
            config=config,
            **self.extra_kwargs,
        )
        for d in data:
            epoch = vars(d)
            step = epoch.pop("up")
            for key, val in epoch.items():
                wandb.log(step=step, data={key: val})

        # Store runtime logs as the main log artifact
        # This will be overwritten in case an unhandled exception occurs
        with (Path(self.wandb.dir) / "output.log").open("w") as f:
            f.write(training_log.logs_str)

    def close(self):
        self.wandb.finish()
