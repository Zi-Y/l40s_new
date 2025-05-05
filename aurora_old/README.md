# Finetuning the Aurora model on CERRA

This repository contains our code to finetune the Aurora model on the CERRA dataset.
It is a fork of the [Microsoft Aurora repository](https://github.com/microsoft/aurora).
The Aurora model is a foundation model for atmospheric forecasting, which we use 
to predict weather in the European region. The original repo has a [documentation website](https://microsoft.github.io/aurora)
, which contains detailed information on how to use the model.

## Getting Started

We use `conda` / `mamba` for development. To install the dependencies, navigate to the repository folder and run:

```bash
mamba env create -f environment.yml
```

To create an environment for aarch64 machines, such as our Grace Hopper servers, run:

```bash
CONDA_OVERRIDE_CUDA="12.6" mamba env create -f environment_aarch64.yaml
```

## Configuration

```bash
configs/
├── data/
│   ├── cerra_debug.yaml
│   ├── cerra.yaml
│   └── dummy.yaml
└── task.yaml
```
### Use of checkpoints
In `task.yaml` specify your ckpt path. The intermediate ckpts are named as `aurora-{model name}-{training phase}-{epoch}-{global step}.ckpt`

Note that it is not possible to continue training from a final checkpoint.


## Training Aurora on CERRA

To finetune the model on 2 GPUs, run:
```python 
OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=2 main.py task.distributed=True dataset=cerra task.phase=finetuning
```



For debugging, we can also use ```dataset=cerra_debug``` on dl09, which will use a cropped
version of the CERRA dataset (512x512 instead of 1069x1069) with only 2 years of data.
When testing somewhere where no CERRA data is available, we can use ```dataset=dummy```,
which will use randomly generated data.

To reduce VRAM usage, we can use ```task.use_activation_checkpointing=True```, which will trade off
some speed for memory usage. Additionally, we can use ```task.use_torch_compile=True``` to compile
the model, which will also reduce memory usage and increase speed. However, due to the compilation
time, this is not recommended for debugging, only for full training runs.

### Logging

Uses optionally  weights and bias for logging metrics and settings, adjust  ```logging.use_wandb``` in task.yaml accordingly
Login in terminal with ```wandb login``` and enter your access key. You find the key here: [Autorize page](https://wandb.ai/authorize)

## Notes
The training code was tested on the gx01 node using 4 GPUs, and it was observed that when the num_workers parameter is set to 8 or higher, the following error occurs:
```RuntimeError: DataLoader worker (pid(s) 3733576) exited unexpectedly.```

## License

See [`LICENSE.txt`](LICENSE.txt).

## FAQ

### How do I setup the repo for local development?

First, install the repository in editable mode and setup `pre-commit`:

```bash
make install
```

To run the tests and print coverage, run

```bash
make test
```

You can then explore the coverage in the browser by opening `htmlcov/index.html`.

To locally build the documentation, run

```bash
make docs
```

To locally view the documentation, open `docs/_build/index.html` in your browser.
