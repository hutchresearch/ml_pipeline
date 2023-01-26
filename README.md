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

- `src/model`, `src/config`, `storage`, `test` dirs.
- `if __name__ == "__main__"` tests.
- Hydra config.
- dataloader, optimizer, criterion, device, state are constructed in main, but passed to an object that runs batches.
- tqdm to track progress.
- debug config flag enables lots breakpoints.
- python type hints.
- a `launch.sh` script to dispatch training.
- a Makefile to install and run stuff.
- automatic linting with the `black` package.
- collate functions!
