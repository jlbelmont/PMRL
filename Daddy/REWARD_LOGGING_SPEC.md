
# REWARD_LOGGING_SPEC.md

## 1. Purpose

Define a **rich reward-logging schema** so that all training runs:

- Are analyzable for stability and exploration quality
- Provide the raw material for:
  - A full NeurIPS-style paper on the RL system
  - A shorter Bayesian progress-monitoring paper

All reward components should be logged in a way that is:

- Machine-readable (CSV / JSONL / TensorBoard / wandb)
- Easy to aggregate at episode, step, and run level
- Compatible with cluster runs (no GUI required)

---

## 2. Reward Components

At each environment step `t`, define:

- `r_env_t` – **extrinsic reward** from the environment
- `r_rnd_t` – RND intrinsic reward
- `r_novel_t` – episodic novelty / visit-count bonus
- `r_bayes_t` – reward derived from Bayesian progress/posteriors
- `r_total_t` – the scalar used for TD learning:

```text
r_total_t = r_env_t + w_rnd * r_rnd_t + w_novel * r_novel_t + w_bayes * r_bayes_t
```

Log also the **weights**:

- `w_rnd`
- `w_novel`
- `w_bayes`

---

## 3. Per-Step Logging Schema

For each transition `(s_t, a_t, r_total_t, s_{t+1}, done_t)` added to the replay buffer, also record the following metadata (either in a sidecar log or extended replay entry):

- `step_global`     – global step index
- `episode_id`
- `env_id`
- `r_env_t`
- `r_rnd_t`
- `r_novel_t`
- `r_bayes_t`
- `r_total_t`
- `milestone_flags` – vector of relevant WRAM milestone flags (e.g., badge bits, story flags)
- `map_id`
- `city_id`
- `quest_id` (if any)
- `posterior_m` – Beta posterior mean for each monitored milestone `m` at time t
- `alarm_score` – progress alarm scalar (see BAYES_PROGRESS_MONITORING.md)

Per-step logging may be sub-sampled (e.g., every N steps) to control log size.

---

## 4. Per-Episode Aggregates

At episode end, log:

- `episode_id`
- `env_id`
- `length_steps`
- `return_total` (sum of `r_env_t`)
- `return_total_with_intrinsic` (sum of `r_total_t`)
- Component returns:
  - `return_env`
  - `return_rnd`
  - `return_novel`
  - `return_bayes`
- `milestones_reached` – list of milestones achieved (badges, story flags)
- `posterior_summary` – for each milestone m:
  - `alpha_m`, `beta_m`
  - posterior mean
  - 95% credible interval bounds
- `alarm_triggered` – boolean / level of Bayesian alarm
- `start_step_global`, `end_step_global`

These aggregates power the **stability plots** and **progress-monitoring figures** in the papers.

---

## 5. File Formats

Preferred options (you can use more than one):

1. **TensorBoard / wandb**:
   - For online training visualization
   - Log scalar summaries per episode and smoothed over steps

2. **CSV / Parquet**:
   - `episodes.csv` for episode-level aggregates
   - `progress.csv` for milestone/posterior trajectories

3. **JSONL**:
   - `events.jsonl` for mixed-events logs, if desired

---

## 6. Visualization Hooks

Downstream, we will want to easily plot:

- Reward component learning curves (per run, averaged over runs)
- Posterior trajectories for each milestone over training
- Alarm lead time vs. actual stagnation events
- Maps of exploration vs. time (requires route occupancy logs from your existing stack)

Therefore, logging_utils should expose simple utilities like:

```python
def log_step_rewards(...)
def log_episode_summary(...)
def flush_episode_logs(...)
```

and **never block** the training loop (asynchronous or buffered logging is preferred).

---

## 7. Paper Alignment

- The **NeurIPS-style full paper** will use:
  - Learning curves
  - Ablations on intrinsic reward components
  - Hierarchical GRU/LSTM/SSM comparisons
  - Stability across seeds and curricula

- The **Bayesian progress-monitoring paper** will use:
  - Beta posteriors per milestone
  - Alarm scores and lead times
  - Comparisons between plain heuristic thresholds vs Bayesian alarms

This spec ensures you have all the **raw signals** needed to support both writing projects.
