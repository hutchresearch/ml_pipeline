# Mimimal Viable Deep Learning Infrastructure

Deep learning pipelines are hard to reason about and difficult to code consistently.

Instead of remembering where to put everything and making a different choice for each project, this repository is an attempt to standardize on good defaults.

Think of it like a mini-pytorch lightening, with all the fory internals exposed for extension and modification.


# Usage

## Install:

Install the conda requirements:

```bash
make install
```

Which is a proxy for calling:

```bash
conda env updates -n ml_pipeline --file environment.yml
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

- `src/model`
- `src/config`
- `data/`
- `test/`
    - pytest: unit testing.
        - good for data shape
        - TODO:
- `docs/`
    - switching projects is easier with these in place
    - organize them

- `**/__init__.py`
    - creates modules out of dir.
    - `import module` works with these.
- `README.md`
    - root level required.
    - can exist inside any dir.
- `environment.yml`
- `Makefile` 
    - to install and run stuff.
    - houses common operations and scripts.
- `launch.sh` 
    - script to dispatch training.

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

## formatting python
- python type hints.
- automatic linting with the `black` package.

## running
- tqdm to track progress.

## architecture
- dataloader, optimizer, criterion, device, state are constructed in main, but passed to an object that runs batches.
