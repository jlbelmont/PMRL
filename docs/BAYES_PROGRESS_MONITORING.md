
# BAYES_PROGRESS_MONITORING.md

## 1. Problem Statement & Motivation

You maintain a large-scale reinforcement learning (RL) project that trains agents to complete **Pokémon Red** end-to-end. The custom RL stack already logs rich telemetry (map visit frequencies, badge unlock times, latent curiosity rewards), but:

- Diagnosing exploration failures
- Detecting regressions in long overnight runs

is still a **manual, qualitative** process.

**Goal:**  
Treat these telemetry streams as **observations of partially observed latent progress variables** and build a **Bayesian progress-monitoring model** that:

1. Quantifies the probability of hitting critical milestones (badges, Elite Four completion) within a given step budget.
2. Conditionally updates the intrinsic-reward schedule to steer the agent away from local optima.

This helps build **trustable, statistically monitored game-playing agents** whose exploration is tracked by formal models instead of ad-hoc eyeballing of curves.

---

## 2. Data Context

- All data comes from your own training harness:
  - ≈ 40 GB of savestates, curiosity signals, route occupancy grids
- You are the primary maintainer of:
  - The environment
  - Telemetry logging
  - WRAM flag decoding

No external datasets are required.

---

## 3. Latent Progress Model (Milestone-Level Beta-Bernoulli)

For each milestone \( m \) (e.g., Boulder Badge, Cascade Badge, Elite Four victory):

- Define a **binary success indicator**:
  - \( Y_{m, \text{episode}} = 1 \) if milestone \( m \) is hit within \( T_m \) steps in that episode
  - \( 0 \) otherwise

- Place a **Beta prior** on the success probability:
  - \( \theta_m \sim \text{Beta}(\alpha_{m,0}, \beta_{m,0}) \)

- After each episode, update:
  - \( \theta_m | \{Y_{m,1}, \dots, Y_{m,n}\} \sim \text{Beta}(\alpha_{m,0} + \sum Y_{m,i},\ \beta_{m,0} + n - \sum Y_{m,i}) \)

**Alarm score:**

- Compute posterior mean \( \mathbb{E}[\theta_m | \text{data}] \)
- If it drifts below a target threshold (e.g., compared to baseline runs or historical performance), raise an **alarm**.

Your telemetry (steps to badge, exploration stats, etc.) determines **how you define \( Y_{m,\text{episode}} \)** and the step budget \( T_m \).

---

## 4. Hierarchical Intrinsic Reward Calibration

Extend to a **hierarchical model** where intrinsic rewards depend on latent progress:

- Let \( r^{\text{intr}}_t \) denote intrinsic reward at time t.
- Model its scale (or rate) as:

  \[
  r^{\text{intr}}_t \sim \text{Gamma}(\kappa_m, \theta_m)
  \]

  where \((\kappa_m, \theta_m)\) are **functions of the milestone posterior**.

- Intuition:
  - Early in training or when progress is poor: **boost curiosity** (larger intrinsic scale).
  - Once a milestone has been reliably achieved: **taper** that curiosity, and shift focus to later milestones.

This creates a **closed loop**:

1. RL agent explores with intrinsic rewards.
2. Telemetry updates milestone posteriors.
3. Posteriors adjust the intrinsic reward schedule.
4. Repeat.

---

## 5. Inference Strategies

You want to compare:

1. **Conjugate Beta updates (online, analytic)**  
   - Already implementable from per-episode binary indicators.

2. **Sequential Monte Carlo (SMC)**  
   - Represent posterior over milestone progress and possibly over hyperparameters.
   - Track multi-modal or non-conjugate structures if you enrich the model.

3. **Gibbs or other MCMC variants**  
   - For offline analysis on stored logs.
   - Useful to compare with SMC and analytic baselines.

**Evaluation metric:**  
Early-warning lead time: how many episodes before a stagnation event does the Bayesian monitor flag the problem, compared to current heuristic threshold-based alerts?

---

## 6. Implementation Hooks in Code

This monitoring lives conceptually in:

- `Daddy/bayes_quests.py`
- `Daddy/logging_utils.py`
- `docs/` analysis notebooks (optional, later)

Implementation steps:

1. **Define milestones**:
   - Map badge flags and key story WRAM flags → milestone IDs \( m \).
   - Attach a step budget \( T_m \) to each.

2. **Episode-level updates**:
   - At episode end, determine for each milestone m:
     - `success_m` (0/1) within \( T_m \) steps
   - Update Beta parameters \( (\alpha_m, \beta_m) \).

3. **Alarm statistics**:
   - Compute posterior means, credible intervals.
   - Derive an alarm score (e.g., probability that \( \theta_m \) is below a target).

4. **Reward coupling**:
   - Use posteriors to adjust:
     - RND scaling
     - Novelty scaling
     - Curriculum choices (savestate sampling)

5. **Logging for papers**:
   - Log per-episode:
     - Beta parameters
     - Posterior means
     - Alarm flags
   - This feeds directly into plots and tables for the Bayesian paper.

---

## 7. Paper Connection

### 7.1 NeurIPS-Style Full Paper

- Context: full RL system (hierarchical GRU/LSTM/SSM, RND, curricula, Bayes progress).
- Sections:
  - Abstract
  - Introduction
  - Related Work (RL exploration, intrinsic motivation, Bayesian monitoring, game agents)
  - Method (architecture + training + Bayes model)
  - Experiments (end-to-end performance, stability, ablations)
  - Discussion

See `PAPER_PLAN_NEURIPS_FULL.md` for a concrete outline.

### 7.2 Short Bayesian Monitoring Paper

- Focus narrowly on the **Bayesian monitor** and its utility.
- Use:
  - Telemetry logs
  - Posterior trajectories
  - Early-warning lead time analysis

See `PAPER_PLAN_BAYES_SHORT.md` for an outline.

---

## 8. Next Steps

1. Implement milestone + posterior tracking in `bayes_quests.py`.
2. Extend `logging_utils.py` and `REWARD_LOGGING_SPEC.md` integration.
3. Start a small notebook / script to:
   - Replay logs
   - Plot posterior vs. time
   - Compute alarm lead times on historical runs.

This document is the **conceptual guide**; the code-level contracts are defined in the other `Daddy/*.md` specs.
