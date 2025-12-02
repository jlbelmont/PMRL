
# DESIGN_SLIM_MODEL.md

# **Slim RL Agent Design Document (Daddy/)**  
### *A Clean, Efficient Replacement for the Old Model in `Final/epsilon/`*  
### *Fully Compatible with `pokemonred_puffer`*

---

## **❗ Important Rules**

### **Do NOT modify any code in:**
- `pokemonred_puffer/` (official puffer environment + wrappers)
- `Final/epsilon/` (my old custom agent / baseline)

These must remain untouched and fully operational.

### **ALL new code must go inside:**
```
Daddy/
```

This will be the **entire codebase for the new slim model**.

---

# **📁 Directory Structure (Required)**

Create this structure under `Daddy/`:

```
Daddy/
  __init__.py

  agent.py                # Main SlimDQN agent class
  networks.py             # CNN + GRU/LSTM + Q-head
  rnd.py                  # Random Network Distillation
  flags.py                # WRAM flag decoding + embedding + objectives
  bayes_quests.py         # Bayesian posterior over quests + flag-likelihoods
  curriculum.py           # Multi-city savestate curriculum (swarming)
  replay_buffer.py        # Uniform or prioritized replay buffer

  logging_utils.py        # steps/sec, parallelization, flag logs, quest logs
  video_utils.py          # mp4 / gif rollout capture

  train_slim.py           # Main training loop using puffer envs
```

---

# **🎯 Objective**

Build a **smaller**, **clean**, **robust**, **value-based off-policy DQN agent** that:

### ✔️ Uses `pokemonred_puffer` for environments  
### ✔️ Works with vectorized envs (multi-env swarming)  
### ✔️ Reads WRAM flag state for objectives + features  
### ✔️ Integrates:
- RND intrinsic rewards  
- Episodic visit-count bonuses  
- Bayesian quest posterior + WRAM flag likelihood modeling  
- Curriculum learning with savestates  
- Full logging (steps/sec, parallelization stats, flag likelihoods, quest posteriors)  
- Video generation (episode rollouts)  

---

# **🧠 Model Requirements (SlimDQN)**

### Input:
- Stacked grayscale frames (72×80, 4–8 frames)
- Structured features (map coords, city id, quest indicators)
- **WRAM flag embeddings** (decoded + compact)

### Architecture:
- **Slim CNN**: 2–3 conv layers, small channel sizes  
- **Recurrent head**: GRU or LSTM, ~128 hidden size  
- **Q-head**: linear output for discrete button actions  
- Total params ideally **2–6M**

### Forward API:
```
forward(obs, structured_features, hidden_state, done_mask)
  → (q_values, new_hidden_state, aux_outputs)
```

---

# **⚙️ Training Loop Requirements (`train_slim.py`)**

### 1. **Vectorized environments**
- Multi-env rollout (“swarming”)
- Each env wrapped by `EpsilonEnv`
- Curriculum chooses savestates per-env

### 2. **DQN core**
- Off-policy learning  
- Replay buffer  
- Target network with periodic sync  
- ε-greedy schedule  

### 3. **Intrinsic motivation**
- RND prediction error reward  
- Episodic novel state bonus  

### 4. **Bayesian quest + flag posterior**
- Maintain posterior over quests/cities  
- Use WRAM flag observations  
- Log posterior snapshots and likelihoods  
- Optional reward shaping from posterior confidence  

### 5. **Curriculum Learning**
- CurriculumManager chooses savestates  
- Promote/demote checkpoints based on success  
- Integrates with vectorized resets  

---

# **📊 Logging Requirements (`logging_utils.py`)**

Log the following:

### **Performance**
- Steps/sec (SPS)
- Total env fps
- Episodes per hour
- Parallelization stats

### **Flags + Quests**
- WRAM flag transitions
- Flag likelihood distributions
- Bayesian posterior over quests/cities/stages

### **Model Internals**
- RND prediction error mean/stdev
- Novelty reward contributions
- Q-value entropy

---

# **🎥 Video Requirements (`video_utils.py`)**

- Capture episode rollouts  
- Render frames → save to mp4 or gif  
- Optionally overlay:
  - Episode reward  
  - WRAM flag indicator text  
  - Quest posterior summary  

---

# **🧪 Debug Scripts**

Add:

```
Daddy/debug_rollout.py
```

Must verify:
- Env → model integration  
- WRAM flag decoding  
- RND output finite  
- Steps/sec normal  
- Video capture works  

---

# **🛠️ Implementation Order for Copilot**
1. Scaffold folder structure  
2. Implement networks.py (CNN + GRU + Q-head)  
3. Implement rnd.py  
4. Implement flags.py  
5. Implement bayes_quests.py  
6. Implement curriculum.py  
7. Implement replay_buffer.py  
8. Implement logging_utils.py  
9. Implement video_utils.py  
10. Implement agent.py (combining everything)  
11. Implement train_slim.py  

---

# **💬 Usage with Copilot**

Inside VSCode:
- Open a file in `Daddy/`
- Tell Copilot:
  > “Use `Daddy/DESIGN_SLIM_MODEL.md` and scaffold the files exactly as specified.”

All future work happens inside `Daddy/` only.

---

