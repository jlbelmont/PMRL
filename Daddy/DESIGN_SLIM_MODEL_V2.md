
# DESIGN_SLIM_MODEL_V2.md

# Slim + Hierarchical RL Agent Design (Daddy/)
### A smaller, structured replacement for the old model in `Final/epsilon/`
### Fully compatible with `pokemonred_puffer`

---

## 1. High-Level Vision

We are building a **hierarchical, multi-timescale value-based RL agent** for Pokémon Red, implemented entirely inside:

```text
/Stimulants_Not_Included/PMR_P_RL_Belmont/Daddy/
```

The agent should:

- Use `pokemonred_puffer` envs as the canonical environment implementation
- Leave `Final/epsilon/` intact for reference/baselines
- Be compact and efficient, but **architecturally rich**:
  - **Short-term module:** GRU
  - **Mid-term module:** LSTM
  - **Long-term module:** SSM (State-Space Model) head
- Support:
  - Multi-env “swarming” vectorized training
  - RND + episodic novelty intrinsic motivation
  - Bayesian quest/flag progress posteriors
  - Curriculum via savestates
  - Full reward logging for downstream analysis and paper writing
  - MP4 + GIF rollout video generation

---

## 2. Directory Structure (Daddy/)

```text
Daddy/
  __init__.py

  agent.py
  networks.py          # CNN + GRU + LSTM + SSM stack + Q-head(s)
  rnd.py
  flags.py
  bayes_quests.py
  curriculum.py
  replay_buffer.py

  logging_utils.py
  video_utils.py

  train_slim.py
  debug_rollout.py

  DESIGN_SLIM_MODEL_V2.md
  REWARD_LOGGING_SPEC.md
  SLIM_AGENT_CLUSTER_README.md
  SLIM_AGENT_VIDEO_SPEC.md

  cluster/
    slurm_gpu.job
    slurm_cpu.job
    env_setup.sh
```

---

## 3. Input & Feature Design

### 3.1 Observations

Each time step provides:

- Stacked grayscale frames: shape `(T_stack, 72, 80)`
- Structured features:
  - Map coordinates, map ID
  - City / route / dungeon IDs
  - Quest indicators / story progress
  - WRAM flags decoded into a compact feature vector (see `flags.py`)

The networks should receive:

```python
obs_frames      # (batch, T_stack, 72, 80)
obs_structured  # (batch, F_struct) – includes WRAM-derived features
```

---

## 4. Network Architecture (networks.py)

The network is **multi-timescale**:

1. **CNN encoder**:
   - 2–3 conv layers with small channels (e.g., 16–32–64)
   - Outputs a per-step embedding: `e_t` of size `D_cnn`

2. **Short-term GRU**:
   - Handles frame-to-frame local dynamics
   - Input: `e_t` concatenated with structured features
   - Hidden size: e.g., 128
   - Produces `h_GRU_t`

3. **Mid-term LSTM**:
   - Operates on a **subsampled or chunked** version of the sequence OR simply consumes `h_GRU_t` at every step.
   - Hidden size: e.g., 128–256
   - Produces `h_LSTM_t`

4. **Long-term SSM block**:
   - A simple, efficient SSM-style module (e.g., 1D convolutional kernel or a simplified linear RNN with learned convolutional kernel) applied over time to capture very long-range structure.
   - Input: `h_LSTM_t`
   - Output: `h_SSM_t`

5. **Heads**:
   - **Q-value head**:
     - Input: `h_SSM_t` (optionally concatenated with `h_GRU_t` and/or `h_LSTM_t`)
     - Output: `Q(a | s_t) ∈ ℝ^{|A|}`
   - Optional auxiliary heads:
     - RND feature projection
     - Flag/quest auxiliary predictors
     - Value baseline / critic head (even though main algorithm is DQN-style)

### Forward API

```python
def forward(
    frames,              # (B, T_stack, 72, 80)
    structured_features, # (B, F_struct)
    state,               # hierarchical recurrent state (GRU, LSTM, SSM)
    done_mask            # (B,) or (B, 1)
) -> (q_values, new_state, aux_outputs)
```

`state` should encapsulate GRU, LSTM, and any SSM hidden variables.

---

## 5. Training Loop (train_slim.py)

Core algorithm: **off-policy Deep Q-Learning** with:

- Replay buffer (uniform or prioritized)
- Target network
- ε-greedy exploration
- TD(0) loss on Q-values
- Intrinsic rewards added to extrinsic reward

### 5.1 Vectorized Environments

- Use `pokemonred_puffer` envs with `EpsilonEnv` wrappers
- Create `num_envs` parallel envs (“swarming”)
- Integrate `CurriculumManager` to choose savestates per-env on reset

### 5.2 Rewards

At each step, store and later log:

- Raw extrinsic env reward
- RND intrinsic reward
- Episodic novelty bonus
- Any Bayesian/posterior-based shaping reward
- Their **sums and breakdowns** (see REWARD_LOGGING_SPEC.md)

---

## 6. Logging & Telemetry

`logging_utils.py` must support:

- Steps per second (SPS)
- Parallelization stats
- Reward component statistics:
  - Moving averages of total reward and each component (extrinsic, RND, novelty, posterior-based)
  - Episode-level aggregates
- Bayesian monitoring stats:
  - Per-milestone Beta parameters (α, β)
  - Posterior means & credible intervals
  - Alarm indicators when progress stalls

All logs must be:

- Cluster-safe (no GUI)
- Persisted to disk (e.g., JSONL, CSV, or TensorBoard / wandb)

See `docs/BAYES_PROGRESS_MONITORING.md` for the conceptual spec and connection to the progress-monitoring paper.

---

## 7. Video Generation

Follow `SLIM_AGENT_VIDEO_SPEC.md` for:

- MP4 + GIF output
- Headless operation (cluster-friendly)
- Optional overlays (rewards, flags, posteriors)

---

## 8. Usage with Copilot/Codex

When asking Copilot to implement or refactor code inside `Daddy/`, always reference this file and the other specs:

> “Read `Daddy/DESIGN_SLIM_MODEL_V2.md`, `Daddy/REWARD_LOGGING_SPEC.md`, `Daddy/SLIM_AGENT_CLUSTER_README.md`, and `Daddy/SLIM_AGENT_VIDEO_SPEC.md` and treat them as the spec. Do not modify `pokemonred_puffer/` or `Final/epsilon/`. Implement or update the requested file accordingly.”

This keeps the implementation aligned with both the **codebase architecture** and the **research goals**.
