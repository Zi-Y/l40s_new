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
OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=2 --master_port=29500 main.py task=finetuning task.distributed=True dataset=cerra
```

Note that if multiple instances of the script are running on the same machine ```master_port``` 
needs to be set to a different value for each instance, otherwise the script will hang.

We currently support four training modesm which can be set with the `task` parameter:
- `finetuning`: Single-step finetuning without rollout
- `rollout_short`: Rollout finetuning where gradients are computed for every step of the rollout at once and the loss is
   averaged over all steps. Only works for short lead times due to memory constraints.
- `rollout_long`: Rollout finetuning where only the last step of the rollout is used to compute the loss. This is more 
   memory efficient and can be used for longer lead times, but it might be slower due to the need to run the model 
   multiple times per gradient step.
- `rollout_long_buffer`: Rollout finetuning with replay buffer. This mode uses a replay buffer to store previous rollout
   states and uses them to compute the loss. This is the most memory efficient mode and can be used for long lead times,
   but it might be less accurate than the other modes because the states in the replay buffer might be outdated.

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

## Inferencing

To create forecasts, set ```task=forecast``` on the command line. Example:

```python
OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=1 main.py 
logging.use_wandb=False dataset=cerra task=forecast task.checkpoint_path={path to checkpoint} task.output_dir={path to output zarr} task.lead_times="[24,48]"
```

This will create forecasts for 24 and 48 hours lead time for the entire validation set.
To change the time range, adjust the time range of the validation set.
The output will be saved in the specified output directory as a zarr file.

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
