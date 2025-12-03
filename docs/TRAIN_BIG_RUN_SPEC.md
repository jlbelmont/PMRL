
# TRAIN_BIG_RUN_SPEC.md
# 🚀 Large-Scale Training Run with Swarming, Savestates, and Terminal Logging

This file is a prompt-spec for Codex / Copilot Max to implement a **full, large-scale training run** for your hierarchical Pokémon Red RL agent.

It assumes:

- Repo root: `/Stimulants_Not_Included/PMR_P_RL_Belmont/`
- Canonical environment: `pokemonred_puffer/` (from `drubinstein/pokemonred_puffer`)
- Old model: `Final/epsilon/` (DO NOT modify)
- New hierarchical agent: `Daddy/` (all new code lives here)
- Curriculum and swarming concepts similar to the original Rubinstein repo (city-/event-based savestates)

The goal:  
**One command** that launches a big training run with:
- Multi-env swarming
- Curriculum savestates (.state files)
- Scaled-up hierarchical model
- Rich terminal logging (so you can literally watch the agent learn)

---

## 1. Training Entry Point: `Daddy/train_big.py`

Create (or refactor) a script:

```text
Daddy/train_big.py
```

### Responsibilities

`train_big.py` must:

1. **Parse CLI arguments** (via `argparse`), including:
   - `--num-envs` (int, default: 32)
   - `--total-steps` (int, e.g. 5_000_000)
   - `--batch-size`
   - `--model-size` (`slim` or `large`, default: `large`)
   - `--run-name` (string, used for log dirs & savestates)
   - `--log-interval` (steps between terminal logs)
   - `--save-interval` (steps between on-disk checkpoints)
   - `--video-interval` (optional, for periodic MP4/GIF rollouts)

2. **Construct the hierarchical agent** from `Daddy/agent.py` and `Daddy/networks.py`:
   - CNN → GRU (short-term) → LSTM (mid-term) → SSM (long-term) → Q-head
   - Use `model-size` to select between:
     - `slim` config (default small)
     - `large` config (scaled up: more channels, bigger hidden sizes)

3. **Create vectorized, swarming environments** from `pokemonred_puffer`:
   - Use multiple parallel `PokemonRedEnv` instances (e.g. `num_envs = 32` or 64).
   - Wrap each env with `EpsilonEnv` and any existing wrappers used in your current pipeline.
   - Integrate curriculum logic via `Daddy/curriculum.py` (see section 3).

4. **Run a full DQN-style training loop**:
   - Replay buffer (size appropriate for large training).
   - Target network with periodic syncs.
   - ε-greedy exploration schedule.
   - TD(0) Q-learning updates.
   - Intrinsic rewards (RND, episodic novelty, Bayesian shaping) combined into `r_total`.
   - Reward logging and metrics as in `REWARD_LOGGING_SPEC.md`.

5. **Be cluster-safe**:
   - No GUI / interactive windows.
   - All logs and videos written to disk.
   - API callable as:

   ```bash
   cd /Stimulants_Not_Included/PMR_P_RL_Belmont
   python -m Daddy.train_big \
       --num-envs 32 \
       --total-steps 5000000 \
       --run-name big_run_01 \
       --log-interval 5000
   ```

---

## 2. Large-Scale Model Configuration

Inside `train_big.py` (or a helper config module), define **model presets**.

Example:

```python
MODEL_CONFIGS = {
    "slim": {
        "cnn_channels": [16, 32, 64],
        "gru_hidden": 128,
        "lstm_hidden": 128,
        "ssm_hidden": 128,
    },
    "large": {
        "cnn_channels": [32, 64, 128],
        "gru_hidden": 256,
        "lstm_hidden": 256,
        "ssm_hidden": 256,
    },
}
```

`train_big.py` should:

- Choose the config based on `--model-size`.
- Pass these parameters into the network construction in `Daddy/networks.py`.

Aim: **Scale up** relative to the slim model, but keep it safe within your GPU memory envelope.

---

## 3. Swarming + Curriculum Savestates (.state Files)

We want to replicate and extend the **Rubinstein-style curriculum**:

- The environment saves `.state` files (RAM + CPU registers + game state) at important game triggers.
- Swarming uses these savestates as **spawn points** for parallel envs.

### 3.1 CurriculumManager (`Daddy/curriculum.py`)

Ensure `Daddy/curriculum.py` exposes a `CurriculumManager` with:

- A registry of savestates:
  - e.g. `{ "pallet_town_start": ".../savestates/pallet_town.state", "pewter_city_gym": ".../savestates/pewter.state", ... }`
- A mechanism for:
  - Sampling initial savestates per env on reset.
  - Updating sampling probabilities based on training progress (milestones, performance, posterior beliefs).

### 3.2 Savestate Saving (Triggers → .state)

Add logic (without breaking puffer) to:

- Watch for **game triggers** via WRAM flags, map IDs, etc., such as:
  - Entering a new city.
  - Obtaining a badge.
  - Beating a gym leader.
  - Reaching a late-game area.

- When such a trigger is observed, call an environment-level function to **save a `.state` file** for that env:
  - Save under a curriculum path, e.g.:

    ```text
    /Stimulants_Not_Included/PMR_P_RL_Belmont/savestates/<run_name>/<milestone_name>_<timestamp>.state
    ```

- Update `CurriculumManager` in memory and optionally on disk (e.g. JSON index) to include the new savestate.

**Codex guidance:**
> When implementing savestate saving, prefer using existing `pokemonred_puffer` save/load functions. Add only minimal adapter code in Daddy/curriculum.py or a small wrapper, and keep any puffer changes backward-compatible.

### 3.3 Swarming From Savestates

When environments **reset**, `CurriculumManager` should:

- Sample a savestate from its registry (possibly conditioned on training progress).
- Load that `.state` into the env instead of always starting from New Game.
- Maintain a mixture of:
  - Early-game savestates (for robustness/rediscovery).
  - Mid-game savestates.
  - Late-game savestates.

This creates **swarming over curriculum**:
- Many parallel agents starting from diverse points in the story.

---

## 4. Terminal Logging (Watch the Agent Learn)

`train_big.py` must provide **rich, human-readable terminal output**.

### 4.1 Requirements

Use `logging` and/or `tqdm`/`rich` to:

- Show a progress bar over `--total-steps`.
- Every `--log-interval` steps, print a structured log line including:

  - `step` / `total_steps`
  - `steps_per_second` (SPS)
  - `epsilon`
  - `recent_avg_episode_length` (sliding window)
  - `recent_avg_r_env`
  - `recent_avg_r_total`
  - Component means:
    - `r_env`
    - `r_rnd`
    - `r_novel`
    - `r_bayes`
  - `episodes_completed`
  - `num_envs`
  - `curriculum_summary` (e.g. fraction of envs spawned from early/mid/late savestates)

- On each episode end, print a brief line, e.g.:

  ```text
  [EP 1342 env=7] len=642  R_env=23.5  R_total=31.8  badges=2  map=PewterCity  savestate_tag=pewter_city_gym  theta_Boulder=0.81
  ```

### 4.2 Dashboard-Style Output (Optional but Nice)

Implement a single-line dashboard that refreshes each log interval, showing:

- Step / total
- SPS
- Avg reward (env + total)
- Epsilon
- Simple summary of milestone posteriors (e.g. Boulder: 0.81, Cascade: 0.42)
- Simple curriculum breakdown (e.g. `spawn: early 45% | mid 35% | late 20%`)

---

## 5. Logging & Saving for Stability Plots

Follow `REWARD_LOGGING_SPEC.md`:

- Log per-episode aggregates to disk:
  - `return_env`, `return_total`, `return_rnd`, `return_novel`, `return_bayes`
  - `milestones_reached`
  - Beta posterior parameters for each milestone
  - `alarm_triggered` flags

- Save **model checkpoints** and **optimizer state** periodically:
  - e.g. in `checkpoints/<run_name>/step_<N>.pt`
  - Include the replay buffer and curriculum metadata if feasible.

---

## 6. Videos (MP4 + GIF) During Long Runs

Using `SLIM_AGENT_VIDEO_SPEC.md`:

- Every `--video-interval` steps (if > 0), run a short evaluation episode and:
  - Record a rollout with overlays (map name, reward components, badge flags).
  - Save both:
    - `rollout_<step>.mp4`
    - `rollout_<step>.gif`

Use `Daddy/video_utils.py` as the implementation module.

---

## 7. UX: One-Command Big Training Run

After `train_big.py` is implemented, I should be able to start a big training run and watch it live:

```bash
cd /Stimulants_Not_Included/PMR_P_RL_Belmont

python -m Daddy.train_big \
    --num-envs 32 \
    --total-steps 5000000 \
    --model-size large \
    --run-name big_run_01 \
    --log-interval 5000 \
    --save-interval 100000 \
    --video-interval 250000
```

This should:

- Launch vectorized swarming training.
- Use curriculum savestates stored as `.state` files.
- Show rich, meaningful terminal logs.
- Produce savestates, checkpoints, logs, and videos on disk.

---

## 8. Codex / Copilot Max Instruction

When you are ready, open `Daddy/train_big.py` and tell Codex:

> “Read TRAIN_BIG_RUN_SPEC.md along with DESIGN_SLIM_MODEL_V2.md, REWARD_LOGGING_SPEC.md, SLIM_AGENT_CLUSTER_README.md, and SLIM_AGENT_VIDEO_SPEC.md. Implement or update `Daddy/train_big.py`, `Daddy/curriculum.py`, and any minimal glue in `Daddy/` so that I can run a large-scale training job with swarming, curriculum savestates (.state files), and rich terminal logging as described.”

This file is your **contract** for the large training run behavior.
