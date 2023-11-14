import csv
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable

import wandb
from translations_parser.data import TrainingEpoch, TrainingLog, ValidationEpoch

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class Publisher(ABC):
    def configure(self, log: TrainingLog):
        ...

    def close(self):
        ...

    @abstractmethod
    def publish_training(epochs: Iterable[TrainingEpoch]):
        ...

    @abstractmethod
    def publish_validation(epochs: Iterable[ValidationEpoch]):
        ...


class CSVExport(Publisher):
    def __init__(self, output_dir):
        assert output_dir.is_dir(), "Output must be a valid directory"
        self.output_dir = output_dir

    def publish_training(self, epochs):
        training_output = self.output_dir / "training.csv"
        if training_output.exists():
            logger.warning(f"Training output file {training_output} exists, skipping.")
            return
        with open(training_output, "w") as f:
            writer = csv.DictWriter(f, fieldnames=TrainingEpoch.__annotations__)
            writer.writeheader()
            for entry in epochs:
                writer.writerow(vars(entry))

    def publish_validation(self, epochs):
        validation_output = self.output_dir / "validation.csv"
        if validation_output.exists():
            logger.warning(f"Validation output file {validation_output} exists, skipping.")
            return
        with open(validation_output, "w") as f:
            writer = csv.DictWriter(f, fieldnames=ValidationEpoch.__annotations__)
            writer.writeheader()
            for entry in epochs:
                writer.writerow(vars(entry))


class WanDB(Publisher):
    def __init__(self, project, **extra_metadata):
        self.project = project
        self.extra_metadata = extra_metadata

    def configure(self, data):
        # Start a new W&B run
        self.wandb = wandb.init(
            project=self.project,
            config={
                **self.extra_metadata,
                **data.configuration,
            },
        )

    def publish_training(self, training_epochs):
        for data in training_epochs:
            wandb.log(step=data.up, data={"training cost": data.cost})

    def publish_validation(self, validation_epochs):
        for data in validation_epochs:
            wandb.log(step=data.up, data={"validation CE (mean words)": data.ce_mean_words})

    def close(self):
        self.wandb.finish()
