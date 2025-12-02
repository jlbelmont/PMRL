
# PAPER_PLAN_NEURIPS_FULL.md

## Provisional Title

**Multi-Timescale Exploration and Bayesian Progress Monitoring in Large-Scale Pokémon Red Reinforcement Learning**

---

## 1. Abstract (Sketch)

- One paragraph summarizing:
  - Large-scale RL for Pokémon Red (end-to-end)
  - Hierarchical GRU/LSTM/SSM agent architecture
  - Intrinsic rewards (RND, novelty) + curricula
  - Bayesian milestone monitoring and intrinsic reward calibration
  - Key empirical findings:
    - Improved stability
    - Better exploration of late-game regions
    - Earlier detection of stagnation / regressions

---

## 2. Introduction

- Motivation: long-horizon, sparse-reward RL in complex games (Pokémon Red).
- Limitations of existing exploration strategies and naive logging.
- Need for:
  - Multi-timescale sequence modeling (short/mid/long term)
  - Statistically grounded progress monitoring.
- Contributions:
  1. A compact yet expressive hierarchical GRU–LSTM–SSM architecture for game-state representation.
  2. An integrated intrinsic-reward stack (RND + novelty + Bayesian shaping).
  3. A Bayesian progress-monitoring model with milestone-level posteriors.
  4. Empirical evaluation showing robustness and earlier stagnation detection.

---

## 3. Related Work

- RL in classic games (Atari, Pokémon, etc.)
- Intrinsic motivation and curiosity-driven exploration.
- Multi-timescale models:
  - RNNs, LSTMs, SSMs, SSD/Mamba-style architectures.
- Bayesian monitoring / anomaly detection / early-warning systems.
- Game telemetry analysis and trustable agents.

---

## 4. Methods

### 4.1 Environment & Task

- Pokémon Red end-to-end.
- Action space, observation stacking.
- WRAM flags, map IDs, quest structure.

### 4.2 Agent Architecture

- CNN feature extractor.
- Short-term GRU.
- Mid-term LSTM.
- Long-term SSM head.
- Q-value readout.
- Implementation details (sizes, activations, etc.).

### 4.3 Intrinsic Rewards

- RND implementation (target + predictor).
- Episodic novelty bonus via visit-counts or hashing.
- Bayesian shaping component linked to milestone posteriors.

### 4.4 Curriculum & Swarming

- Parallel envs.
- Savestate curricula across cities and late-game regions.
- CurriculumManager’s policies.

### 4.5 Bayesian Progress Monitoring

- Milestones, Beta-Bernoulli model.
- Extensions to hierarchical Gamma–Beta for intrinsic reward scaling.
- Alarm scores and thresholds.

---

## 5. Experimental Setup

- Cluster hardware (ENGR cluster).
- Training hyperparameters.
- Baselines:
  - Old model in `Final/epsilon/`.
  - Non-hierarchical DQN.
  - No-Bayes vs Bayes shaping.
- Datasets:
  - Training logs (40 GB of telemetry).
  - Number of seeds and runs.

---

## 6. Results

### 6.1 Learning Curves & Stability

- Return vs. environment steps.
- Variance across seeds.
- Comparison to baselines.

### 6.2 Exploration Metrics

- Map coverage.
- Badge acquisition times.
- Elite Four success rates.

### 6.3 Bayesian Monitoring Performance

- Posterior trajectories for milestones.
- Early-warning lead time vs heuristic thresholds.
- Examples of runs recovered by adaptive shaping.

### 6.4 Ablations

- Remove SSM → GRU/LSTM only.
- Remove GRU/LSTM → SSM-only.
- Remove Bayes shaping.
- Freeze curricula.

---

## 7. Discussion

- Insights on multi-timescale modeling in RL.
- Practical lessons for cluster-scale RL experiments.
- Limitations:
  - Complexity vs interpretability.
  - Data requirements.
- Future work:
  - More general game domains.
  - Richer SSM architectures (e.g., Mamba-style).

---

## 8. Conclusion

- Summarize the value of:
  - Hierarchical sequence models + intrinsic rewards + Bayesian monitoring.
- Emphasize **trustable RL agents** with statistically monitored progress.

---

## 9. Appendices / Supplement

- Additional architecture diagrams.
- Hyperparameter tables.
- Extra ablation studies.
- Implementation details (SLURM scripts, logging schema).
