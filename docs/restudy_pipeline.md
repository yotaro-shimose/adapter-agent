# Restudy Pipeline

The **restudy** pipeline closes the loop on RL training: it mines the tasks a
finished RL run could *not* solve (`c == 0` at the last `rl_step`), uses Gemini
to plan what library knowledge was missing, generates SFT-ready QRA triples,
buckets each QRA by self-learnability (via pass@k), and feeds the result back
into a small KSFT → KRL → TaskRL training pass.

Five scripts make up one round. Each section below lists what to run, what it
reads, and what it writes — including the manual checkpoint-URI hand-offs
between the training stages.

---

## 0. Prerequisites

- A completed RL training run with rollouts persisted to `simple_rl_rollouts`
  (i.e. the run wrote a `simple_train_id` row + per-task rollouts).
- The `restudy_numrs2_qra` Prisma cache id is free to reset (the pipeline
  clears it at start; pass `reset_target_caches=False` to keep history).
- Tinker SDK + Postgres + cloudrun runtimes configured (same as the rest of
  the project).

---

## 1. Mine failures + generate QRAs

[scripts/restudy_pipeline.py](../scripts/restudy_pipeline.py) reads zero-success
tasks from a named RL run and runs the
`plan → investigate → augment → variant-verify → reason → persist` chain.

**Edit before running** (constants near the top of the file):

```python
RL_FAILURES_TRAIN_ID = "continue_rl_task_numrs2_20260510_010510"  # source RL run
MAX_FAILURES: int | None = None  # int caps the failure count for fast iteration
```

**Knobs** (`FailureWorkloadConfig`, `FailureStageConfig`):

- `items_per_failure_limit` (default 3) — how many plan items to keep per failure
- `variants_per_item` (default 3) — paraphrases per verified investigation
- `planner_max_turns` (default 15) — tool-using planner budget (currently unused; one-shot planner is the active path)
- `investigation_max_turns` (default 10) — investigator budget
- `variant_verify_max_turns` (default 4) — variant verifier budget

**Run**:

```bash
uv run scripts/restudy_pipeline.py > logs/restudy_pipeline.log 2>&1
```

**Outputs** (Prisma cache ids):

- `restudy_numrs2_inv` — raw verified investigations
- `restudy_numrs2_aug` — paraphrased variants
- `restudy_numrs2_qra` — final SFT-ready (Q, R, A) triples ← **this is what KSFT / KRL train on**

---

## 2. Pass@k routing

[scripts/passatk_restudy.py](../scripts/passatk_restudy.py) evaluates each QRA
in `restudy_numrs2_qra` against the current TaskRL checkpoint (16 samples
each) and writes a routing CSV that splits each task into one of three
buckets:

| Bucket | Condition | Meaning |
|---|---|---|
| `SFT` | `c == 0` | model can't solve at all — needs SFT |
| `RL` | `1 ≤ c < 9` (= < 60% of 16) | partial signal — RL can amplify |
| `Proficient` | `c/n ≥ 60%` | already solved — drop |

**Run**:

```bash
uv run scripts/passatk_restudy.py > logs/passatk_restudy.log 2>&1
```

**Output**:

- `logs/passatk/restudy_self_learnability.csv` — `task_idx, instruction, n_samples, success_count, success_rate, bucket`

The KRL recipe reads this CSV to filter its seed pool.

---

## 3. Restudy KSFT (1 epoch on restudy QRAs)

[scripts/run_continue_rl.py](../scripts/run_continue_rl.py) drives all three
training stages — switch `CONFIG = ...` to pick which one.

**Before running**, in `run_continue_rl.py`:

```python
# Resume from the canonical TaskRL output (= the checkpoint passatk evaluated).
_NUMRS2_TASK_RL_CHECKPOINT_BASE = "tinker://be9e6178-...:train:0"
_NUMRS2_TASK_RL_CHECKPOINT_NAME = "rl_0040"

CONFIG = NUMRS2_RESTUDY_KSFT_RECIPE
```

**Knobs** (`NUMRS2_RESTUDY_KSFT_RECIPE.pipeline_config.sft`):

- `epochs=1, batch_size=32, learning_rate=7e-5` — matches RL stage lr so the
  SFT→RL transition stays smooth. Previous default `1e-4` caused catastrophic
  forgetting on `gh_archive_eval`.

**Run**:

```bash
uv run scripts/run_continue_rl.py > logs/restudy_ksft.log 2>&1
```

**Hand-off**: grep the log for the saved checkpoint URI and update
`_NUMRS2_RESTUDY_KSFT_CHECKPOINT_BASE` in the script:

```bash
grep "Saved checkpoints.*init_sft" logs/restudy_ksft.log
# → 'state_path': 'tinker://<NEW_UUID>:train:0/weights/init_sft'
```

```python
_NUMRS2_RESTUDY_KSFT_CHECKPOINT_BASE = "tinker://<NEW_UUID>:train:0"
```

**Sanity check before moving on**: open WandB and confirm
`eval/gh_archive_eval/success_ratio` didn't collapse (should stay within a few
points of the pre-KSFT baseline). If it tanks (e.g. < 5%), reduce SFT
intensity (fewer epochs, lower lr) and rerun.

---

## 4. Restudy KRL (RL on SFT+RL bucket + replay mix)

```python
CONFIG = NUMRS2_RESTUDY_KRL_RECIPE
```

**What this run does**:

- Seeds two suites into the RL pool:
  - `restudy_numrs2_qra_rl` — the SFT+RL bucket of `restudy_numrs2_qra`,
    filtered via the routing CSV.
  - `pipeline_v2_qra_numrs2_replay` — the broad-knowledge KRL pool. Acts as
    a forgetting brake.
- Mixes them per-rollout via `RLConfig.suite_mix_weights` (default 50/50).
- Drops any task that hits 100% success in a single rollout group (no more
  re-queue — saves rollout compute on already-mastered tasks).
- Logs per-suite metrics (`rollout/<suite>/mean_reward`,
  `rollout/<suite>/any_correct_ratio`, `rollout/<suite>/n_groups`) to WandB
  alongside the aggregate ones.

**Knobs** (`NUMRS2_RESTUDY_KRL_RECIPE.pipeline_config.rl`):

- `max_iterations=30` (current) or `num_passes=N` — choose one
- `batch_size=16, learning_rate=7e-5`
- `suite_mix_weights={"restudy_numrs2_qra_rl": 1.0, "pipeline_v2_qra_numrs2_replay": 1.0}`
  — adjust ratio to shift restudy-vs-replay emphasis
  (e.g. `{"restudy": 4.0, "replay": 1.0}` = 80/20)

**Run**:

```bash
uv run scripts/run_continue_rl.py > logs/restudy_krl.log 2>&1
```

**Hand-off**:

```bash
grep "Saved checkpoints" logs/restudy_krl.log | tail -1
# → 'state_path': 'tinker://<NEW_UUID>:train:0/weights/rl_NNNN'
```

```python
_NUMRS2_RESTUDY_KRL_CHECKPOINT_BASE = "tinker://<NEW_UUID>:train:0"
_NUMRS2_RESTUDY_KRL_CHECKPOINT_NAME = "rl_NNNN"  # the final step
```

---

## 5. Restudy TaskRL (final RL on gh_archive[0:150])

```python
CONFIG = NUMRS2_RESTUDY_TASK_RL_RECIPE
```

This is the standard Task RL stage, but resumed from the restudy KRL output.
Trains on `gh_archive[0:150]`, evals on `gh_archive[150:200]` — same evaluation
slice as the canonical `NUMRS2_TASK_RL_RECIPE` so the two are comparable.

**Run**:

```bash
uv run scripts/run_continue_rl.py > logs/restudy_task_rl.log 2>&1
```

The resulting `tinker://...rl_NNNN` is the new "current TaskRL" — if you
plan another restudy round, point `_NUMRS2_TASK_RL_CHECKPOINT_BASE`
(step 3) at this and start over from step 1 with the new `simple_train_id`.

---

---

## Running the same pipeline for hisab

The pipeline supports both numrs2 and hisab via a `LIBRARY` switch at the top
of `restudy_pipeline.py` and `passatk_restudy.py`. Flipping to hisab points
the loaders at the hisab failure-source RL run, the hisab cache ids, and
LibrarySpec.hisab(). Everything downstream (qra_pipeline stages, passatk,
recipes) works unchanged.

### Differences from the numrs2 lineage

| | numrs2 | hisab |
|---|---|---|
| Failure source | `continue_rl_task_numrs2_20260510_010510` | `continue_rl_task_hisab_from_qra_v2_20260507_120638` |
| Resume base (KSFT) | `be9e6178/rl_0040` (NUMRS2_TASK_RL_RECIPE) | `ca15e826/rl_0030` (HISAB_TASK_RL_FROM_DECOMPOSED_RECIPE) |
| QRA cache | `restudy_numrs2_qra` | `restudy_hisab_qra` |
| Routing CSV | `restudy_self_learnability.csv` | `restudy_hisab_self_learnability.csv` |
| Replay pool (KRL) | `pipeline_v2_qra_numrs2` | `pipeline_v2_qra` |

### KSFT catastrophic-forgetting fix (hisab-only)

A naive 1-epoch SFT on `restudy_hisab_qra` alone collapsed gh_archive_eval
from baseline to **1.5%** — out-of-band (Gemini-generated) SFT data pulls
the model toward the data source's distribution. Fix: anchor the SFT update
with **on-policy** replay — the model's own past successful rollouts from
TaskRL + KRL runs, sampled to maximize task diversity + step recency
(latest-success-per-task across both runs).

Implementation: [`load_rl_rollout_replay_suite`](../adapter_agent/simple_internalizer/sft_qra_loaders.py)
pulls `simplerlrollout` rows where `success=True`, keeps one per task at
the highest `rl_step`, and random-samples down to a configured count.

The hisab KSFT recipe wires three sources:

```python
sft_sources=[
    load_sft_cache_suite(name="restudy_hisab_qra", ...),           # 208 QRAs
    load_rl_rollout_replay_suite(
        name="hisab_taskrl_replay",
        simple_train_ids=["continue_rl_task_hisab_from_qra_v2_20260507_120638"],
        take_n=115,
    ),
    load_rl_rollout_replay_suite(
        name="hisab_krl_replay",
        simple_train_ids=["continue_rl_hisab_from_qra_v2_20260506_235537"],
        take_n=1341,
    ),
]
# 208 + 115 + 1341 = 1664 QRAs → ~12.5%/87.5% restudy/replay per batch
```

Result trajectory (post-KSFT gh_archive_eval as a function of replay weight):

| restudy : replay | post-KSFT | notes |
|---|---|---|
| 1 : 0 (no replay) | 1.5% | catastrophic |
| 1 : 3 | 7.0% | partial recovery |
| 1 : 7 | 8.0% | plateaued |

The plateau is acceptable — KRL replay-mix RL recovers it (see below).

### Hisab lineage results

| Stage | gh_archive_eval | Checkpoint |
|---|---|---|
| canonical hisab TaskRL | — (baseline) | `tinker://ca15e826-...:train:0/weights/rl_0030` |
| Restudy KSFT (1:7 on-policy replay) | 8.0% | `tinker://7fd46c37-...:train:0/weights/init_sft` |
| Restudy KRL (30 iter, 50/50 mix) | 18.0% | `tinker://1e72a865-...:train:0/weights/rl_0030` |
| Restudy TaskRL step 25 (in-pipeline) | 31.0% | `tinker://1dcb5948-...:train:0/weights/rl_0025` |
| **Restudy TaskRL rl_0030 (`run_eval`)** | **36.0%** | `tinker://1dcb5948-...:train:0/weights/rl_0030` |

Same eval slice (gh_archive[150:200]) as canonical HISAB_TASK_RL recipes
so the lineage is directly comparable.

---

## Reference: scripts and what they own

| Script | Role |
|---|---|
| [scripts/restudy_planner.py](../scripts/restudy_planner.py) | `FailedAttempt`, one-shot planner (`plan_one_shot`), planner-task prompt builder |
| [scripts/restudy_pipeline.py](../scripts/restudy_pipeline.py) | Driver: load failures → plan → investigate → QRA persistence |
| [scripts/passatk_restudy.py](../scripts/passatk_restudy.py) | pass@k routing of the QRA cache against TaskRL ckpt → CSV |
| [scripts/run_continue_rl.py](../scripts/run_continue_rl.py) | KSFT / KRL / Task RL recipes (switch via `CONFIG = ...`) |

## Reference: shared infrastructure

| Module | What it provides |
|---|---|
| [adapter_agent/hierarchical/qra_pipeline.py](../adapter_agent/hierarchical/qra_pipeline.py) | Stage runners (`run_investigate`, `augment_verify_reason`, persistence) — shared with `study2_pipeline.py` |
| [adapter_agent/simple_internalizer/seed_loaders.py](../adapter_agent/simple_internalizer/seed_loaders.py) | `load_sft_cache_seed_suite_factory_bucket_filtered` — bucket-filtered seed loader for KRL |
| [adapter_agent/simple_internalizer/rl_worker_pool.py](../adapter_agent/simple_internalizer/rl_worker_pool.py) | Per-suite weighted sampling + proficient-task auto-drop |
| [adapter_agent/simple_internalizer/pipeline.py](../adapter_agent/simple_internalizer/pipeline.py) | `RLBatchState._flush_window` emits per-suite rollout metrics |
