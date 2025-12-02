
# PMR_P_RL_Belmont – Repo Bootstrap & Layout

## 1. Goal

Create a **new Git repository** called `PMR_P_RL_Belmont` that:

- Is based on the code from `https://github.com/drubinstein/pokemonred_puffer`
- Lives at the path:

  ```
  /Stimulants_Not_Included/PMR_P_RL_Belmont/
  ```

- Contains:
  - The original `pokemonred_puffer` code (unmodified baseline)
  - Your existing experimental model under `Final/`
  - The new slim / hierarchical model under `Daddy/`

All new work for the slim/hierarchical agent must go in `Daddy/`.  
The old agent remains in `Final/`.  
The puffer env code remains under `pokemonred_puffer/` and should only be changed in **very small, well-documented, backward-compatible ways** if absolutely necessary.

---

## 2. Directory Layout (Target)

Inside `/Stimulants_Not_Included/PMR_P_RL_Belmont/`:

```text
PMR_P_RL_Belmont/
  pokemonred_puffer/        # from drubinstein/pokemonred_puffer (baseline env + agent)
  Final/
    epsilon/                # existing older model – do NOT modify
  Daddy/                    # new slim/hierarchical RL stack
    DESIGN_SLIM_MODEL_V2.md
    REWARD_LOGGING_SPEC.md
    SLIM_AGENT_CLUSTER_README.md
    SLIM_AGENT_VIDEO_SPEC.md
    README.md
    __init__.py
    agent.py
    networks.py
    rnd.py
    flags.py
    bayes_quests.py
    curriculum.py
    replay_buffer.py
    logging_utils.py
    video_utils.py
    train_slim.py
    debug_rollout.py
    cluster/
      slurm_gpu.job
      slurm_cpu.job
      env_setup.sh
  docs/
    BAYES_PROGRESS_MONITORING.md
    PAPER_PLAN_NEURIPS_FULL.md
    PAPER_PLAN_BAYES_SHORT.md
```

You can add more subfolders (e.g., `configs/`, `notebooks/`) as needed.

---

## 3. Bootstrap Steps (Manual / Once)

From `/Stimulants_Not_Included/`:

```bash
# 1) Clone original puffer repo as the base
git clone https://github.com/drubinstein/pokemonred_puffer PMR_P_RL_Belmont
cd PMR_P_RL_Belmont

# 2) Make folders for Final/ and Daddy/
mkdir -p Final/epsilon
mkdir -p Daddy/cluster
mkdir -p docs

# (You will copy your existing Final/epsilon code into here manually.)

# 3) Initialize new repo identity (optional if you want a separate GitHub project)
git remote remove origin   # detach from original repo
git remote add origin git@github.com:<your-username>/PMR_P_RL_Belmont.git
git branch -M main
```

After this, `PMR_P_RL_Belmont` is **your** repo, with `pokemonred_puffer` as an internal module/folder.

---

## 4. Ground Rules for Code Changes

1. **Do not modify:**
   - `Final/epsilon/`
   - Any existing training scripts that you rely on for the old model

2. **Minimize changes to:**
   - `pokemonred_puffer/`
   - If you must modify env/wrapper code, make:
     - Small, documented changes
     - Backward-compatible (no breaking old agents)
     - Prefer adding optional flags / wrappers rather than changing behavior

3. **All new research / logging / hierarchical model work goes into:**
   - `Daddy/` (code)
   - `docs/` (papers & methodology)

---

## 5. GitHub Notes

Once the layout is in place and initial files are added:

```bash
git add .
git commit -m "Bootstrap PMR_P_RL_Belmont from pokemonred_puffer and add Daddy/docs structure"
git push -u origin main
```

You now have a clean, **research-grade** repo ready for Codex/Copilot to extend.
