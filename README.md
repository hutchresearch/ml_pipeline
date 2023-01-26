# Mimimal Viable Deep Learning Infrastructure

Deep learning pipelines are hard to reason about and difficult to code consistently.

Instead of remembering where to put everything and making a different choice for each project, this repository is an attempt to standardize on good defaults.

Think of it like a mini-pytorch lightening, with all the fory internals exposed for extension and modification.

This project lives here: [https://github.com/publicmatt.com/ml_pipeline](https://github.com/publicmatt.com/ml_pipeline).


# Usage

```bash
make help # lists available options.
```

## Install:

Install the conda requirements:

```bash
make install
```

## Data:

Download mnist data from PJReadie's website:

```bash
make data
```

## Run:

Run the code on MNIST with the following command:

```bash
make run
```

# Tutorial

The motivation for building a template for deep learning pipelines is this: deep learning is hard enough without every code baase being a little different.

Especially in a research lab, standardizing on a few components makes switching between projects easier.

In this template, you'll see the following:

## directory structure
```
.
├── README.md
├── environment.yml
├── launch.sh
├── Makefile
├── data
│   ├── mnist_test.csv
│   └── mnist_train.csv
├── docs
│   └── 2023-01-26.md
├── src
│   ├── config
│   │   └── main.yaml
│   ├── data
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── collate.py
│   │   └── dataset.py
│   ├── eval.py
│   ├── __init__.py
│   ├── model
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── cnn.py
│   │   └── linear.py
│   ├── pipeline
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── logger.py
│   │   ├── runner.py
│   │   └── utils.py
│   ├── sample.py
│   └── train.py
└── test
    ├── __init__.py
    └── test_pipeline.py

8 directories, 25 files

```

## what and why?

- `environment.yml`
    - hutch research has standardized on conda
    - here's a good tutorial on getting that setup: [seth email](emailto:bassetis@wwu.edu)
- `launch.sh` or `Makefile`
    - to install and run stuff.
    - houses common operations and scripts.
    - `launch.sh` to dispatch training.
- `README.md`
    - explain the project and how to run it.
    - list authors.
    - list resources that new collaborators might need.
    - root level dir.
    - can exist inside any dir.
    - reads nicely on github.com.
- `docs/`
    - switching projects is easier with these in place.
    - organize them by meeting, or weekly agenda.
    - generally collection of markdown files.
- `test/`
    - TODO
    - pytest: unit testing.
    - good for data shape. not sure what else.
- `data/`
    - raw data
    - do not commit these to repo generally.
        - `echo "*.csv" >> data/.gitignore`
- `__init__.py`
    - creates modules out of dir.
    - `import module` works b/c of these.
- `src/model/`
    - if you have a large project, you might have multiple architectures/models.
    - small projects might just have `model/VGG.py` or `model/3d_unet.py`.
- `src/config`
    - based on hydra python package.
    - quickly change run variables and hyperparameters.
- `src/pipeline`
    - where the magic happens.
    - `train.py` creates all the objects, hands them off to runner for batching, monitors each epoch.

## testing
- `if __name__ == "__main__"`.
    - good way to test things
- enables lots breakpoints.

## config
- Hydra config.
    - quickly experiment with hyperparameters
    - good way to define env. variables
        - lr, workers, batch_size
        - debug

## data
- collate functions!
- datasets.
- dataloader.

## formatting python
- python type hints.
- automatic linting with the `black` package.

## running
- tqdm to track progress.
- wandb for logging.

## architecture
- dataloader, optimizer, criterion, device, state are constructed in main, but passed to an object that runs batches.

