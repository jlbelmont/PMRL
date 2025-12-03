
# VISUALS_FOR_RL_TRAINING.md
# 📊 Visual Blueprints for Pokémon Red RL Training
### High‑impact visualizations for papers, slideshows, dashboards, and real‑time monitoring

This file contains a structured list of **ideal, publication‑grade visualizations** you can generate from your Pokémon Red RL training pipeline. Each section explains:

1. **What the visualization is**
2. **Why it is scientifically interesting**
3. **What it communicates in a paper or talk**
4. **A Codex‑ready implementation prompt**

Use this to build a full visualization module in `Daddy/visuals/`.

---

# 1. 🗺️ Global Map Occupancy Heatmap
### “Where has the agent explored across the world?”

**Description:**  
Render a stitched map of Kanto and overlay a heatmap of visit frequency for each (x,y) location.

**Why interesting:**  
Shows spatial exploration behavior, escape from local minima, curriculum effects.

**Paper usage:**  
Exploration quality figure; curiosity ablation.

**Codex prompt:**  
> “Create `plot_global_map_heatmap(occupancy_grid, map_asset)` using matplotlib. Overlay tile visitation counts on a stitched Kanto map. Save PNG + MP4.”

---

# 2. 🌀 Swarm Trajectory Visualization  
### “How 32–128 vectorized envs move during training”

**Description:**  
At each global step, plot the position of each parallel environment over the map.

**Why interesting:**  
Shows exploration diversity and curriculum spread.

**Paper usage:**  
Stability and robustness of multi-env swarming.

**Codex prompt:**  
> “Implement `plot_swarm_positions(env_positions, map_asset)` rendering each env's location as a colored dot, optionally animated to MP4.”

---

# 3. 🛣️ Per‑Episode Trajectory Overlay  
### “Detailed path taken by the agent in one episode”

**Description:**  
Draw a polyline on the regional map for a single episode.

**Why interesting:**  
Concrete policy behavior.

**Paper usage:**  
Side‑by‑side ‘before training’ vs ‘after training’ trajectories.

**Codex prompt:**  
> “Implement `draw_episode_path(path_coords, map_asset)` exporting to MP4 or PNG.”

---

# 4. 🧭 Bayesian Milestone Posterior Timelines  
### “Posterior confidence in reaching badges / milestones”

**Description:**  
Plot Beta posterior means & credible intervals for each milestone over training.

**Why interesting:**  
Supports your Bayesian progress‑monitoring paper.

**Paper usage:**  
Posterior trajectories demonstrating monitoring effectiveness.

**Codex prompt:**  
> “Implement `plot_milestone_posteriors(posterior_history)` with shaded 95% intervals.”

---

# 5. 🔥 Exploration Frontier Map  
### “Where is the agent still curious?”

**Description:**  
Heatmap of RND prediction error or novelty score per tile.

**Why interesting:**  
Shows agent’s active learning target zones.

**Paper usage:**  
Illustration of intrinsic reward shaping.

**Codex prompt:**  
> “Implement `plot_exploration_frontier(novelty_grid, map_asset)`.”

---

# 6. 🎞️ Annotated Rollout Video (MP4 + GIF)
### “High‑impact visual for talks”

**Overlay options:**
- Current map & coordinates  
- Action selected  
- Q‑value summary  
- Badge flags  
- Posterior milestone bars  
- Reward breakdown  

**Why interesting:**  
Super persuasive demo of agent behavior.

**Paper usage:**  
Supplemental videos for publication.

**Codex prompt:**  
> “Extend `video_utils.py` to overlay HUD elements on frames before encoding to MP4/GIF.”

---

# 7. 🧩 Representation Space Embedding (UMAP / PCA)
### “What states look like in embedding space”

**Description:**  
Project GRU/LSTM/SSM embeddings to 2D.

**Why interesting:**  
Shows hierarchical representation learning.

**Paper usage:**  
Demonstrates learned structure across cities, gyms, etc.

**Codex prompt:**  
> “Implement `plot_state_embeddings(embeddings, labels)` using PCA→UMAP.”

---

# 8. 📉 Action‑Entropy Timeline  
### “Is the agent exploring or exploiting?”

**Description:**  
Entropy of action distribution (or Q‑value softmax) over time.

**Why interesting:**  
Correlates with learning stability, collapse, or over‑exploration.

**Paper usage:**  
Figure in training dynamics section.

**Codex prompt:**  
> “Add `plot_action_entropy(entropy_series)`.”

---

# 9. 🔀 Curriculum Pathway Graph
### “How the curriculum or savestates are structured”

**Description:**  
Nodes = savestate clusters. Edges = transition / sampling frequency.

**Why interesting:**  
Explains curriculum curriculum dynamics visually.

**Paper usage:**  
Ablation comparing curriculum vs no-curriculum.

**Codex prompt:**  
> “Implement `plot_curriculum_graph(graph_data)` using networkx.”

---

# 10. 🧱 Replay Buffer Composition Analysis  
### “What does the agent actually learn from?”

**Description:**  
Histogram of state categories in replay buffer:
- Overworld  
- Towns  
- Gyms  
- Menus  
- Battles  
- PokeCenters  

**Why interesting:**  
Shows dataset distribution & training bias.

**Paper usage:**  
Strong diagnostic plot.

**Codex prompt:**  
> “Implement `plot_buffer_state_distribution(buffer_stats)`.”

---

# 11. 🧬 Multi‑Timescale Dynamics Panel  
### “GRU, LSTM, and SSM internals side‑by‑side”

**Description:**  
Line charts of:
- GRU activations  
- LSTM cell states  
- SSM convolutional responses  

**Why interesting:**  
Explains interplay between short/medium/long temporal scales.

**Paper usage:**  
Architectural understanding figure.

**Codex prompt:**  
> “Implement `plot_recurrent_dynamics(hidden_logs)`.”

---

# 12. 🎇 Final Training Summary Poster  
### “One figure to rule them all”

Combine:
- Global heatmap  
- Posterior timeline  
- Exploration frontier  
- Badge timing heatmap  
- Action entropy timeline  

**Paper usage:**  
Introductory or concluding illustration.

**Codex prompt:**  
> “Implement `create_training_summary_poster(figures...)` via matplotlib gridspec.”

---

# Quick commands: plot returns + Bayes posteriors from a run

1) Activate env and set paths:
```
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/pokemonred_puffer:$PYTHONPATH"
```
2) Plot the latest run in `runs/` (returns + Bayes if `progress.csv` exists):
```
python scripts/plot_latest.py
```
3) Or specify a run dir explicitly:
```
python scripts/plot_latest.py --run runs/short10k
```
Outputs:
- `plot_returns.png` (episode returns or events-based rolling `r_total`)
- `plot_bayes.png` (posterior means per milestone)

# 📌 How to Use This With Codex Max

Paste into Copilot Chat:

> “Read VISUALS_FOR_RL_TRAINING.md. Create a new folder `Daddy/visuals/` and generate function stubs for each visualization with TODO comments based on the descriptions.”

This will produce a full visualization suite scaffolded for your agent.
