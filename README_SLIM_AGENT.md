# Slim Pokémon Red RL Agent — GitHub Repository README

# 🎮 **Slim Pokémon Red RL Agent (DQN + WRAM Flags + RND + Curriculum)**  
### **A clean, modular replacement for `Final/epsilon/`, fully compatible with `pokemonred_puffer`**

---

## ⭐ Overview

This repository implements a **slim, efficient Deep Q-Network agent** for Pokémon Red, designed to run on the canonical `drubinstein/pokemonred_puffer` environment without modifying it.

Your old model lives in:

```
Final/epsilon/
```

This new system lives entirely in:

```
Daddy/
```

Both can coexist without interference.

---

# 📁 **Repository Structure**

```
.
├── pokemonred_puffer/            # Do NOT modify (official env)
├── Final/
│   └── epsilon/                  # Old model (leave untouched)
├── Daddy/                        # New slim agent (this project)
│   ├── DESIGN_SLIM_MODEL.md
│   ├── README.md
│   ├── __init__.py
│   ├── agent.py
│   ├── networks.py
│   ├── rnd.py
│   ├── flags.py
│   ├── bayes_quests.py
│   ├── curriculum.py
│   ├── replay_buffer.py
│   ├── logging_utils.py
│   ├── video_utils.py
│   ├── train_slim.py
│   ├── debug_rollout.py
└── ...
```

---

# 🚀 **Project Goals**

The slim agent must:

### ✔️ Stay fully compatible with `pokemonred_puffer`  
### ✔️ Not modify `Final/epsilon/`  
### ✔️ Implement a clean DQN architecture  
### ✔️ Use WRAM flags as objectives + state features  
### ✔️ Support vectorized envs (multi-env swarming)  
### ✔️ Include:
- Random Network Distillation  
- Episodic novelty reward  
- Bayesian quest/flag posterior model  
- Swarmed curriculum learning  
- Comprehensive logging (steps/sec, flag likelihoods, posterior dumps)  
- Video recording of rollouts  

---

# 🔧 **SlimDQN Architecture Requirements**

### Inputs:
- 72×80 grayscale frame stacks (4–8 frames)
- Structured map/goal features
- **WRAM flag embeddings**

### Core architecture:
- **Slim CNN** (2–3 simple conv layers)
- **GRU or LSTM (128 hidden)**
- **Linear Q-value head**

Total params target: **2–6 million**.

---

# ⚙️ Training Loop (train_slim.py)

### Must include:
- Vectorized Pokémon Red envs  
- Replay buffer  
- Target network  
- ε-greedy schedule  
- RND intrinsic reward  
- Episodic novelty bonus  
- Curriculum learning  
- Bayesian quest/flag posterior  
- Logging + video recording  

---

# 📊 Logging Requirements

- Steps/sec (SPS)
- Env parallelization stats
- RND prediction error distribution
- Novelty reward logs
- WRAM flag likelihoods
- Bayesian posterior over quests/cities
- Q-value entropy
- Episode reward summaries

---

# 🎥 Video Requirements

- Capture rollouts to MP4 or GIF  
- Optional overlays:
  - Episode reward  
  - WRAM flag states  
  - Quest posterior snapshot  

---

# 🧪 Debug Script

`debug_rollout.py` must:
- Run a brief vectorized rollout  
- Log steps/sec  
- Verify WRAM flag decoding  
- Verify RND works (no NaNs)  
- Optionally save a short video  

---

# 🛠️ How to Use This With Copilot/Codex

Open VSCode → open a file in `Daddy/` → open Copilot Chat → send:

```
Read Daddy/DESIGN_SLIM_MODEL.md and treat it as the specification.
Follow it exactly and scaffold the new slim agent inside Daddy/.
Do NOT modify pokemonred_puffer/ or Final/epsilon/.
Start by generating networks.py and agent.py.
```

Then follow up:

```
Now create rnd.py exactly as described.
```

```
Now create flags.py exactly as described.
```

etc… until all files match the spec.

---

# ✔️ This repo is now ready for GitHub

Just commit:

```
git add .
git commit -m "Add slim Pokémon Red RL agent architecture and design spec"
git push
```

Enjoy building your slim RL system!
