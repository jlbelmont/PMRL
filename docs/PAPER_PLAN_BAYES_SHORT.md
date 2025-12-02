
# PAPER_PLAN_BAYES_SHORT.md

## Provisional Title

**Bayesian Progress Monitoring for Long-Horizon Game Agents: A Case Study in Pokémon Red**

---

## 1. Abstract (Sketch)

- Focus: Bayesian monitoring layer, not the entire RL stack.
- Problem: Hard to know when long-horizon RL agents are “stuck” or regressing.
- Approach:
  - Treat telemetry as observations of latent milestone success.
  - Use Beta-Bernoulli and hierarchical Gamma–Beta models.
  - Generate alarms and adapt intrinsic rewards.
- Findings:
  - Earlier stagnation detection.
  - More efficient use of training time.

---

## 2. Introduction

- Context: monitoring large RL jobs (hours/days).
- Pain point: manual dashboard watching.
- Proposal: Bayesian monitor that:
  - Quantifies probability of milestone success.
  - Issues alarms when progress deteriorates.
- Pokémon Red RL as a compelling testbed.

---

## 3. Methodology

### 3.1 Data & Telemetry

- Telemetry signals:
  - Badge events.
  - Elite Four victories.
  - Episode lengths.
  - Map occupancy grids.
  - Intrinsic reward statistics.

### 3.2 Latent Progress Model

- Milestones m and budgets T_m.
- Beta priors on success probabilities.
- Conjugate posteriors from per-episode binary outcomes.

### 3.3 Hierarchical Intrinsic Reward Model

- Gamma-distributed intrinsic reward scales.
- Hyperparameters conditioned on milestone posteriors.

### 3.4 Alarm Logic

- Define thresholds on posterior means / credible intervals.
- Alarm when probability of meeting target falls below a bound.
- Optionally, define multi-level alarms (warning vs critical).

---

## 4. Experiments

- Use logs from multiple long training runs:
  - With and without Bayesian monitor.
  - Possibly replay them offline for retrospective evaluation.

- Metrics:
  - Time-to-alarm relative to actual stagnation.
  - False positive / false negative rate.
  - Effect on learning curves when the monitor is allowed to adjust reward scaling.

---

## 5. Results

- Plots of:
  - Posterior trajectories vs episodes.
  - Alarm points vs actual failures.
  - Distribution of alarm lead times.
- Case studies of:
  - Runs where monitor correctly anticipates collapse.
  - Runs where heuristics miss problems that Bayes catches.

---

## 6. Discussion & Conclusion

- Practical utility of Bayesian progress monitors.
- Interaction with existing RL debugging workflows.
- Directions:
  - More complex latent models.
  - Integration with multi-game benchmarks.
- Short punchy conclusion: statistical, not just visual, monitoring.

---
