# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The conda env is named `robodiff` (created from `conda_environment.yaml`; macOS dev uses `conda_environment_macos.yaml`, real-robot uses `conda_environment_real.yaml`). Activate it before running anything: `conda activate robodiff`. The env pins Python 3.9, PyTorch 1.12.1+cu116, gym==0.21.0, hydra-core 1.2, and diffusers 0.11.1 — the codebase depends on these specific versions (e.g. `gym==0.21.0` API, old `diffusers` scheduler signatures), so do not casually upgrade.

The package is installed via `pip install -e .` (see `setup.py`); imports use `from diffusion_policy....`. The inner `diffusion_policy/` directory is the actual Python package; the outer one is the repo root.

## Common commands

```bash
# Single-seed training (hydra config under diffusion_policy/config/)
python train.py --config-name=train_diffusion_unet_image_workspace task=pusht_image training.seed=42 training.device=cuda:0

# Using a downloaded experiment config file (yaml at repo root)
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42

# Multi-seed training via Ray (start cluster first: `ray start --head --num-gpus=N`)
python ray_train_multirun.py --config-name=<workspace> --seeds=42,43,44 --monitor_key=test/mean_score -- <hydra overrides>

# Evaluation from a checkpoint
python eval.py --checkpoint <path>/latest.ckpt --output_dir <out> --device cuda:0

# Aggregate metrics across train_0/train_1/... runs in a multi-seed dir
python multirun_metrics.py -i data/outputs/<run_dir> -k test/mean_score

# Real robot
python demo_real_robot.py -o data/demo_pusht_real --robot_ip <ip>
python eval_real_robot.py -i <ckpt> -o <out> --robot_ip <ip>

# Tests (pytest)
pytest tests/                          # all
pytest tests/test_replay_buffer.py     # single file
pytest tests/test_replay_buffer.py::test_name  # single test
```

Hydra controls all config composition. `task=<name>` swaps the `task` subtree with `diffusion_policy/config/task/<name>.yaml`. Outputs land in `data/outputs/yyyy.mm.dd/hh.mm.ss_<name>_<task_name>/` (configured under `hydra.run.dir` in the workspace YAML).

## Architecture

The codebase enforces an `O(N+M)` split between `N` tasks and `M` methods — they are implemented independently behind a fixed interface, with deliberate copy-paste rather than cross-cutting abstractions. When adding a task or method, copy the closest sibling rather than refactor shared code.

**Workspace is the entry point.** `train.py` reads a top-level YAML (e.g. `config/train_diffusion_unet_image_workspace.yaml`), instantiates the class at `cfg._target_` (a subclass of `diffusion_policy.workspace.base_workspace.BaseWorkspace`), and calls `.run()`. The Workspace's `run()` method contains the entire train/eval loop. Checkpointing happens at the Workspace level: `BaseWorkspace.save_checkpoint` introspects `self.__dict__` and persists anything with `state_dict`/`load_state_dict`, plus attributes listed in `include_keys`. Anything that should NOT be in the checkpoint must live as a local variable inside `run()`, not as `self.*`.

**The task/policy split:**
- Task side: `dataset/<task>_dataset.py` (subclass of `BaseLowdimDataset`/`BaseImageDataset`, exposes `get_normalizer()`), `env_runner/<task>_runner.py` (subclass of `BaseLowdimRunner`/`BaseImageRunner`, exposes `run(policy) -> dict` for wandb), `config/task/<task>.yaml` (wires both via `_target_`, and declares `shape_meta`).
- Policy side: `policy/<method>_policy.py` (subclass of `BaseLowdimPolicy`/`BaseImagePolicy`, implements `predict_action(obs_dict) -> action_dict` and usually `compute_loss(batch)`; handles its own normalization via `set_normalizer`/`LinearNormalizer`), `workspace/train_<method>_workspace.py`, `config/train_<method>_workspace.yaml`.

**Horizon terminology** (paper → code): observation horizon `To` = `n_obs_steps`; action horizon `Ta` = `n_action_steps`; prediction horizon `T` = `horizon`. Datasets return `(To, ...)` obs and `(Ta, Da)` actions; policies see batched `(B, To, ...)` / `(B, Ta, Da)`.

**Normalization** is the most common bug source. Each Policy owns a `LinearNormalizer` (parameters saved in the checkpoint); it is built from the Dataset's `get_normalizer()` at training start. When debugging suspect mismatches, print the per-key `scale`/`bias` vectors.

**Data layout — `ReplayBuffer`** (`diffusion_policy/common/replay_buffer.py`) is the zarr-backed (or numpy-backed) demonstration store. Each `data/<field>` array is all episodes concatenated along the time axis; `meta/episode_ends` gives episode boundary indices. Image datasets often hold the full buffer in RAM with Jpeg2000 compression (see `diffusion_policy/codecs/imagecodecs_numcodecs.py`). Sample windows are drawn by `SequenceSampler` (`common/sampler.py`); episode-boundary padding for `To`/`Ta` is handled there — read it before writing custom sampling.

**Vectorized evaluation** uses `diffusion_policy/gym_util/async_vector_env.py` (modified `gym.vector.AsyncVectorEnv`). Subprocesses are forked, so any env that creates an OpenGL context at construction (robosuite especially) will inherit a broken context and segfault. Provide a `dummy_env_fn` that constructs the env without OpenGL when this matters.

**Real-robot path is async, not gym-style.** `diffusion_policy/real_world/real_env.py` splits `gym.step` into `get_obs()` (non-blocking read of the latest frame from a `SharedMemoryRingBuffer`) and `exec_actions(actions, timestamps)` (enqueue into `RTDEInterpolationController` via `SharedMemoryQueue`, returns immediately). Per-sensor subprocesses (e.g. `SingleRealsense`) own the SDK pipeline and write into their ring buffer. The shared-memory primitives in `diffusion_policy/shared_memory/` avoid pickle/queue overhead.

## Adding a task or method

To add a task: copy `dataset/pusht_image_dataset.py`, `env_runner/pusht_image_runner.py`, `config/task/pusht_image.yaml`. Update `shape_meta` to your input/output shapes and point `_target_`s at the new classes. Run with `task=<your_task>`.

To add a method: copy `workspace/train_diffusion_unet_image_workspace.py`, `policy/diffusion_unet_image_policy.py`, `config/train_diffusion_unet_image_workspace.yaml`. The workspace YAML's top-level `_target_` must point at the new Workspace class.
