# Mimimal Viable Deep Learning Infrastructure

Deep learning pipelines are hard to reason about and difficult to code consistently.

Instead of remembering where to put everything and making a different choice for each project, this repository is an attempt to standardize on good defaults.

Think of it like a mini-pytorch lightening, with all the fory internals exposed for extension and modification.


## Usage

### Install:

Install the conda requirements:

```bash
make install
```

Which is a proxy for calling:

```bash
conda env updates -n ml_pipeline --file environment.yml
```

### Run:

Run the code on MNIST with the following command:

```bash
make run
```

