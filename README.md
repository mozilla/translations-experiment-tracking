# translations-experiment-tracking

Track and extract data from the training system of Firefox Translations.

Logs are extracted from [Marian](https://marian-nmt.github.io/) training tasks, running in Task Cluster.

This POC works offline, using a text log sample within the `samples` directory. It outputs an instance of the `TrainingLog` dataclass with the following attributes:
  * `info`: Marian information as a dict
  * `configuration` Runtime configuration as a dict
  * `training` List of Training dataclass instances:
    * `epoch`
    * `up`
    * `sen`
    * `cost`
    * `time`
    * `rate`
    * `gnorm`
  * `validation` List of `Validation` dataclass instances:
    * `epoch`
    * `up`
    * `chrf`
    * `ce_mean_words`
    * `bleu_detok`
  * `logs` as a dict of log lines, indexed by their header (e.g. marian, data, memory)

## Install and run the package

On a virtual environment, you can install the package using pip:
```sh
$ pip install .
```

Run the parser with the local sample:
```sh
$ parse_training_logs
```

## Development

On a virtual environment, you can install the package using pip:
A developer may want to install the package in editable mode (i.e install from the local path directly):
```sh
$ pip install -e .
```

Pre-commit rules are automatically run once pre-commits hooks have been installed:
```sh
$ pip install pre-commit
$ pre-commit install
$ pre-commit run -a # Run pre-commit once
```
