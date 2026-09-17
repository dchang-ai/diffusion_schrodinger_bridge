# CLAUDE.md

Reference implementation of **Diffusion Schrödinger Bridge** (De Bortoli, Thornton, Heng, Doucet, 2021) — IPF/IPFP solved with score-matching networks. Research code: no tests, no linter, no packaging.

## Environment

Python 3.12, PyTorch 2.x. `requirements.txt` holds the dependency list (lower bounds, no upper pins); `conda.yaml` just creates a 3.12 env and defers to it.

```bash
python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
# or: conda env create -f conda.yaml && conda activate bridge
```

Data download (CelebA / Stacked MNIST) is a separate step: `python data.py --data celeba --data_dir ./data/`.

## Running

Hydra drives everything; config groups are selected on the command line:

```bash
python main.py dataset=2d model=Basic num_steps=20 num_iter=5000
python main.py dataset=stackedmnist model=UNET num_steps=30 num_iter=5000 data_dir=<abs path to data>
python main.py dataset=celeba      model=UNET num_steps=50 num_iter=5000 data_dir=<abs path to data>
```

2d runs on CPU; image datasets assume CUDA (`device: cuda`, `dataparallel: True`) and were developed on 2 high-memory V100s. Reduce `cache_npar` / `num_cache_batches` when the cache OOMs — the cache, not the batch, is usually what blows up memory.

**Hydra changes the working directory** to `./experiments/<date>/<name>/<overrides>/<time>/` (see `conf/job.yaml`). Everything the runner writes — `checkpoints/`, `im/`, `gif/`, `logs/`, `run.log` — lands there, and any *relative* `data_dir` is resolved against that run dir, not the repo root. Pass `data_dir` as an absolute path.

Resume: `checkpoint_run=True checkpoint_it=<n> checkpoint_pass=<f|b> checkpoint_f=<path> checkpoint_b=<path>` (plus `sample_checkpoint_f/b` for the EMA copies).

## Config layout (`conf/`)

- `config.yaml` — primary config, defaults to `dataset: 2d` + `model: Basic`. `conf/mnist.yaml` is an alternative primary config (`python main.py --config-name mnist`).
- `dataset/{2d,stackedmnist,celeba}.yaml` — carry *most* hyperparameters (training, diffusion schedule, device, logging), not just data settings. All groups are `# @package _global_`, so their keys land at the top level of `args`.
- `model/{Basic,UNET}.yaml` — `Basic` is just a tag; `UNET` also sets the `model.*` subtree.

Gotchas:
- Keys defined directly in the primary config (`mean_match`, `num_cache_batches`, `cache_refresh_stride`, …) **override** the same keys coming from a default-list group. E.g. `mean_match: False` in `dataset/2d.yaml` is shadowed by `mean_match: True` in `config.yaml`. Override on the CLI when you want the group's value.
- `mean_final` / `var_final` are **strings `eval()`'d** in `config_getters.get_datasets` with `torch` in scope (e.g. `1 * torch.ones([${data.channels}, ...])`). They are only used when both `adaptive_mean` and `final_adaptive` are False; otherwise the prior moments are estimated from a 100-sample batch of the data.

## Architecture

`main.py` → `IPFSequential(args).train()` (`bridge/runners/ipf.py`).

**Outer loop (IPF/IPFP).** `train()` runs `n_ipf` iterations; each iteration calls `ipf_step('b', n)` then `ipf_step('f', n)`. Two networks live in `self.net = ModuleDict({'f':…, 'b':…})`, each with its own optimizer and EMA helper.

**Inner loop (score matching).** `ipf_step` trains one direction for `num_iter` steps on an MSE loss against trajectories produced by the *other* direction's frozen (EMA) network. With `use_prev_net: False` the network is rebuilt from scratch at every IPF step.

**Trajectory cache** (`bridge/data/cacheloader.py`). Sampling is expensive, so `CacheLoader` pre-rolls `num_cache_batches × cache_npar` full Langevin trajectories into a tensor of `(x, out, step)` triples and is refreshed every `cache_refresh_stride` training iterations. Special case: at `n == 1` in the backward pass there is no trained forward net yet, so the cache uses `langevin.record_init_langevin` — the analytic reference OU process toward the Gaussian prior.

**Diffusion** (`bridge/langevin.py`). `gammas` is a symmetric schedule (`gamma_min → gamma_max` over `num_steps//2`, then mirrored); `T = sum(gammas)`. The network is conditioned on **cumulative time** `torch.cumsum(gammas)`, not on the step index, and `ipf.py` feeds it `T - steps_expanded` to invert time.

**`mean_match` couples two files.** When True the network predicts the next mean directly (`pred = net(x, t) - x` in `ipf.py`, `t_old = net(x, t)` in `langevin.py`); when False it predicts the drift. Both files must agree — change it only via config, never in one place.

**Distributed.** HuggingFace `accelerate` wraps nets/optimizers/dataloaders (`Accelerator`, `self.accelerator.backward(loss)`); `dataparallel: True` additionally wraps in `torch.nn.DataParallel`, which is why `save_step` unwraps `.module` before `state_dict()`.

**Registry pattern.** `bridge/runners/config_getters.py` maps config tags → objects: `get_models`, `get_datasets`, `get_plotter`, `get_logger`, `get_optimizers`. To add a dataset or model, add the tag constant + branch there and a matching `conf/dataset/*.yaml` or `conf/model/*.yaml` (and export the model from `bridge/models/__init__.py`). UNet `channel_mult` is hardcoded per image size in `get_models` — a new resolution needs an entry or it raises.

**Other modules.** `bridge/models/basic/` = time-conditioned MLP for 2d; `bridge/models/unet/` = ADM-style UNet for images. `bridge/runners/plotters.py`: `TwoDPlotter` (scatter/trajectory/registration PNGs + GIFs) vs `ImPlotter`, gated by `plot_level` and `gif_stride`. `bridge/data/__init__.py` also contains an unused legacy `get_dataset`/`get_dataloader` path (CIFAR10/LSUN) not reached by `config_getters`.

## Python 3.12 / modern-stack notes

Migrated off the original Python 3.8 / torch 1.8 stack. Things that were adapted and should not regress:

- `@hydra.main(version_base=None, ...)` in `main.py`, `hydra.job.chdir: True` in `conf/job.yaml` (Hydra 1.2+ stopped chdir'ing by default), and an explicit trailing `- _self_` in both primary configs to keep the pre-1.1 override ordering.
- `Accelerator(mixed_precision='no', ...)` — accelerate dropped the `fp16=` flag.
- `torch.load(..., weights_only=True)` everywhere (torch 2.6 flipped the default); checkpoints also load with `map_location`.
- `torch._six.string_classes` (gone in torch 2.0) → `(str, os.PathLike)` in `bridge/data/vision.py`.
- `from torch.utils.model_zoo import tqdm` (no longer re-exported) → `from tqdm import tqdm`, and `six.moves.urllib` → stdlib `urllib.request`, in `bridge/data/utils.py`.
- pandas `delim_whitespace=True` (removed in pandas 3.0) → `sep=r"\s+"` in `bridge/data/celeba.py`.
- `bridge/runners/logger.py` targets pytorch-lightning 2.x: `NeptuneLogger` takes `project=` (not `project_name=`) and is imported lazily; `CSVLogger.save()` flushes *and* clears its own buffer, so no manual buffer reset.

Two fixes were needed to run on macOS at all (both also correct on Linux):

- `worker_init_fn` was a closure inside `build_dataloaders`; macOS/Windows default to the `spawn` start method, which can't pickle it, so any `num_workers > 0` crashed. It is now the module-level `WorkerInitFn` class in `bridge/runners/ipf.py`.
- `gammas` came from numpy as float64, and MPS has no float64 support. It is now built as float32 (Langevin cast it to float anyway).

## Running on Apple silicon

`device: cuda` in the image configs resolves to **MPS** through accelerate — leave it as `cuda`, do not switch it to `cpu` (CPU is ~8x slower here). `dataparallel: True` is a silent no-op without CUDA. The `pin_memory` warning on MPS is harmless.

Measured on an M5 Pro (24 GB unified memory), current stack:

| config | training | cache rebuild |
|---|---|---|
| 2d / Basic, bs 512, CPU | ~104 it/s | ~0.06 s |
| stackedmnist / UNET 64ch, bs 128, MPS | ~2.5 it/s | ~30 s per `num_cache_batches` (cache_npar 1000, 30 steps) |
| stackedmnist / UNET 32ch, bs 128, MPS | ~5.4 it/s | ~12 s |
| celeba / UNET 64ch, 64x64, MPS | ~78 img/s fwd+bwd | — |

A 2d plot event (`save_step` on TwoDPlotter) costs ~3 s. At the README's settings the 2d example finishes in ~35 min; stackedmnist takes ~40 h and celeba well over a week, and celeba at `batch_size: 128` exceeds 24 GB and degrades into swap (step time climbing 2 s -> 60 s). Scale `n_ipf`, `num_iter`, `model.num_channels` and `num_cache_batches` down for local runs.

The UNet requires `model.num_channels` to be a multiple of 32 (`GroupNorm32`) — torch now raises instead of silently accepting it.

CelebA's Google-Drive download (`bridge/data/utils.py`) is independently broken by Google's interstitial; fetch the archive manually into `<data_dir>/celeba/` if you need it.
