# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reimplementation of **DreamerV3**, a model-based RL algorithm that learns a world model and trains an actor-critic policy from imagined trajectories. It supports a wide range of environments (Atari, DMC, Crafter, DMLab, Minecraft, ProcGen, BSuite, LocoNav) with a fixed set of hyperparameters.

The stack: JAX + [ninjax](https://github.com/danijar/ninjax) for neural nets, [portal](https://github.com/danijar/portal) for multiprocess communication, [elements](https://github.com/danijar/elements) for utilities/logging/checkpointing, and [optax](https://github.com/deepmind/optax) for optimization.

## Commands

**Install dependencies (requires JAX installed first):**
```sh
pip install jax[cuda12]==0.4.33  # or appropriate JAX version for your hardware
pip install -U -r requirements.txt
```

**Run training (single process):**
```sh
python dreamerv3/main.py \
  --logdir ~/logdir/{timestamp} \
  --configs crafter \
  --run.train_ratio 32
```

**Run with debug config (fast, uses CPU, small model):**
```sh
python dreamerv3/main.py \
  --configs crafter debug \
  --logdir ~/logdir/debug
```

**Run on CPU explicitly:**
```sh
python dreamerv3/main.py --configs crafter --jax.platform cpu
```

**Resume training (same command, same logdir):**
```sh
python dreamerv3/main.py --logdir ~/logdir/existing_run --configs crafter
```

**View metrics:**
```sh
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

**Run tests:**
```sh
cd embodied/tests && python -m pytest test_train.py -v
cd embodied/tests && python -m pytest test_replay.py -v
cd embodied/tests && python -m pytest test_driver.py -v
```

**Run a single test:**
```sh
cd embodied/tests && python -m pytest test_train.py::TestTrain::test_run_loop -v
```

**Docker:**
```sh
docker build -f Dockerfile -t img .
docker run -it --rm -v ~/logdir/docker:/logdir img \
  python main.py --logdir /logdir/{timestamp} --configs minecraft debug
```

## ManiSkill-HAB Integration

The file `embodied/envs/mshab.py` bridges ManiSkill-HAB into DreamerV3. Only three files were touched:

| File | Change |
|------|--------|
| `embodied/envs/mshab.py` | New wrapper (sole integration point) |
| `dreamerv3/main.py` | One line: `'mshab': 'embodied.envs.mshab:MSHab'` in the `ctor` dict |
| `dreamerv3/configs.yaml` | New `mshab` config block |

**Installation** (separate conda env recommended):
```sh
conda create -n mshab python=3.9 && conda activate mshab
git clone https://github.com/haosulab/ManiSkill.git -b mshab --single-branch
pip install -e ManiSkill
pip install -e /path/to/mshab/repo
for dataset in ycb ReplicaCAD ReplicaCADRearrange; do
  python -m mani_skill.utils.download_asset "$dataset"
done
pip install jax[cuda12]==0.4.33 && pip install -U -r requirements.txt
```

**Run training:**
```sh
python dreamerv3/main.py \
  --logdir ~/logdir/{timestamp} \
  --configs mshab \
  --task mshab_tidy_house_pick
```

**Available tasks** (override `--task`):
- `mshab_tidy_house_pick` / `mshab_tidy_house_place`
- `mshab_prepare_groceries_pick` / `mshab_prepare_groceries_place`
- `mshab_set_table_pick` / `mshab_set_table_place` / `mshab_set_table_open` / `mshab_set_table_close`

**How the wrapper works:**

- **Gymnasium → embodied.Env**: MS-HAB returns a 5-tuple `(obs, rew, term, trunc, info)`; the wrapper converts this to DreamerV3's dict-based `step()` interface.
- **Tensor → numpy**: MS-HAB returns PyTorch GPU tensors; `_to_np()` does `.detach().cpu().float().numpy()`.
- **Batch dimension**: MS-HAB always batches even with `num_envs=1`; actions are unsqueezed `[None]` before `step()`, obs are indexed `[0]` on return.
- **Depth → uint8 HWC**: Depth images arrive as `(1, C, H, W)` float32 (metres). The wrapper transposes to `(H, W, C)` and clips/scales to `[0, 255]` uint8 so DreamerV3's CNN encoder picks them up as image inputs.
- **Termination**: `terminated` (task success/failure) → `is_terminal=True`; `terminated OR truncated` → `is_last=True`.
- **No frame-stacking**: DreamerV3's RSSM handles temporal context; `FrameStack` is intentionally omitted.

**GPU parallelism tradeoff**: Each env wrapper instance uses `num_envs=1` inside ManiSkill. DreamerV3's `Driver` spawns `run.envs` subprocesses each with their own GPU context. Keep `run.envs` small (1–4) to avoid GPU OOM. For production, a custom run mode using larger `num_envs` and a batched driver would be more efficient.

## Architecture

### Top-level structure

- `dreamerv3/` — the DreamerV3 agent implementation
  - `main.py` — entry point; parses config, selects run script, wires together agent/replay/env/logger factories
  - `agent.py` — `Agent` class (world model + actor-critic training), `imag_loss`, `repl_loss`, `lambda_return`
  - `rssm.py` — `RSSM`, `Encoder`, `Decoder` modules (the core world model networks)
  - `configs.yaml` — all hyperparameters; task-specific configs override `defaults`

- `embodied/` — the RL infrastructure library
  - `core/` — `Agent`/`Env` base classes, `Driver`, `Replay`, `Selectors`, `Streams`, `Wrappers`, limiters
  - `jax/` — JAX-specific base `Agent`, `MLPHead`, `Optimizer`, `Normalize`, `SlowModel`, layer definitions (`nets.py`), output distributions (`outs.py`)
  - `envs/` — environment adapters (Atari, DMC, Crafter, DMLab, Minecraft, etc.)
  - `run/` — training loop scripts: `train.py` (single-process), `train_eval.py`, `eval_only.py`, `parallel.py` (multi-process)
  - `tests/` — pytest test suite using a dummy env and a `TestAgent` stub

### Key data flow

1. **`main.py`** selects a run script based on `config.script` (`train`, `train_eval`, `eval_only`, `parallel`, `parallel_env`, `parallel_replay`).

2. **Single-process training** (`embodied/run/train.py`): creates a `Driver` that manages parallel env processes, attaches `on_step` callbacks for replay insertion and training, and calls `agent.policy` / `agent.train` / `agent.report`.

3. **Parallel training** (`embodied/run/parallel.py`): spawns separate processes for the agent (actor+learner threads), environment workers, replay buffer, and logger — all communicating via `portal` RPC servers. The actor serves policy calls to envs; the learner pulls batches from replay.

4. **`dreamerv3/agent.py`** extends `embodied.jax.Agent`. It holds the encoder (`enc`), RSSM dynamics (`dyn`), decoder (`dec`), reward head (`rew`), continuation head (`con`), policy (`pol`), value (`val`), and slow value target (`slowval`). All parameters share a single `Optimizer`.

5. **RSSM** (`dreamerv3/rssm.py`) maintains a recurrent state `(deter, stoch)`. `observe` processes real trajectories; `imagine` rolls out from latent starts using the policy for actor-critic training.

6. **Replay** (`embodied/core/replay.py`) stores experiences as chunked sequences on disk and in memory, supports prioritized/recency/uniform sampling mixtures, and provides `replay_context` entries for recurrent carry re-initialization.

### Config system

All hyperparameters live in `dreamerv3/configs.yaml` under named blocks. Multiple blocks are layered in order via `--configs block1 block2 ...`. Config values can also be overridden directly on the command line (e.g., `--run.train_ratio 64`). The `debug` config block is the go-to for fast iteration — it shrinks the model and batch sizes and forces CPU.

Size presets (`size1m` through `size400m`) override `rssm`, `depth`, and `units` fields via regex patterns that match nested config keys.

### Observation/action conventions

- `obs_space` keys starting with `log/` are excluded from the encoder and treated as logging-only scalars.
- `act_space` always contains a `reset` key (boolean) used by `Driver` and env wrappers but excluded from the agent's action space.
- Environments are wrapped with `NormalizeAction`, `UnifyDtypes`, `CheckSpaces`, and `ClipAction` in `make_env → wrap_env`.

### Checkpointing

`elements.Checkpoint` saves/loads agent weights, replay buffer, and logger step to `<logdir>/ckpt/`. Resuming is automatic — running the same command with the same `--logdir` continues from the last checkpoint.
