You are working in the repo:

  /Stimulants_Not_Included/PMR_P_RL_Belmont/

The hierarchical agent and training harness live in:

  Daddy/

Use the spec files already in this repo (DESIGN_SLIM_MODEL_V2.md, REWARD_LOGGING_SPEC.md, SLIM_AGENT_CLUSTER_README.md, SLIM_AGENT_VIDEO_SPEC.md, etc.) as the source of truth.

I want you to set up a **full, scaled-up training run** of the hierarchical DQN agent with **rich terminal logging** so I can visually watch the agent learn in real time.

CONSTRAINTS:
- Do NOT modify Final/epsilon/ at all.
- Keep any changes to pokemonred_puffer/ minimal and backward-compatible.
- All new training logic and logging lives in Daddy/.

====================================================
A) TRAINING ENTRYPOINT
====================================================

1. In Daddy/, create or update a script:

   Daddy/train_big.py

   This script should:

   - Import the hierarchical agent and network from Daddy/agent.py and Daddy/networks.py.
   - Use a **“large” model configuration**:
       * More channels in the CNN
       * Larger GRU/LSTM/SSM hidden sizes
       * More layers where appropriate
     (Scale up relative to the default slim config, but still within GPU memory limits.)

   - Use a **larger training setup**:
       * num_envs: e.g. 32–64 parallel envs (“swarming”)
       * replay buffer: larger capacity
       * longer total training steps

   - Accept CLI arguments via argparse, for example:
       * --num-envs
       * --total-steps
       * --batch-size
       * --run-name
       * --log-interval (steps between terminal logs)
       * --eval-interval (optional eval episodes)

   - Call a main() so I can run:

     python -m Daddy.train_big --num-envs 32 --total-steps 5_000_000 --run-name big_run_01

====================================================
B) TERMINAL LOGGING REQUIREMENTS
====================================================

I want **live terminal feedback** during training, not just TensorBoard/wandb.

Please:

1. Use either:
   - Python’s logging module with a nice, readable format, and/or
   - tqdm / rich progress bars for step-level progress.

2. Every N steps (controlled by --log-interval), print a log line that includes at minimum:

   - Global step
   - Steps per second (SPS)
   - Current epsilon
   - Recent average episode length (last K episodes)
   - Recent average extrinsic reward
   - Recent average total reward (with intrinsic)
   - Decomposition of reward components:
       * mean r_env
       * mean r_rnd
       * mean r_novel
       * mean r_bayes
   - Number of episodes completed
   - Number of active envs

3. On every episode completion, log a short summary line, e.g.:

   [EP 1342 env=7] len=642  R_env=23.5  R_total=31.8  badges=2  map=BrockGym  theta_Boulder=0.81

4. Optionally add a **single-line “dashboard”** that refreshes in place in the terminal (tqdm / rich) showing:

   - Global step / total steps
   - SPS
   - Avg reward (sliding window)
   - Current curriculum level / savestate distribution overview (e.g. % of envs in early/mid/late game)
   - A compact string for flags (e.g. “Badges: 110010”)

====================================================
C) CONFIG FOR “LARGE” MODEL
====================================================

Add a configuration block or a simple Python dictionary for a “large” model preset, for example in Daddy/train_big.py or a small config module:

- CNN:
  * channels [32, 64, 128] instead of [16, 32, 64]
- GRU hidden size: 256
- LSTM hidden size: 256–512
- SSM block: larger state dimension or kernel size

Tie this to a flag --model-size {slim, large} with default “large” in train_big.py.

====================================================
D) HOOKS INTO EXISTING LOGGING & REWARD SPEC
====================================================

Use REWARD_LOGGING_SPEC.md as a contract:

- Make sure the training loop populates per-episode aggregates:
    * return_env
    * return_total
    * return_rnd
    * return_novel
    * return_bayes
    * milestones_reached
    * posterior_summary for milestones (optional in terminal, mandatory in logs)

- For terminal output, show a **compressed** subset (just enough to see learning), and write detailed metrics to disk (CSV/JSONL/TensorBoard) as already specified.

====================================================
E) USER EXPERIENCE
====================================================

The final goal: I should be able to run one command in my terminal and watch the agent learn:

  cd /Stimulants_Not_Included/PMR_P_RL_Belmont
  python -m Daddy.train_big \
      --num-envs 32 \
      --total-steps 5_000_000 \
      --run-name big_run_01 \
      --log-interval 5000

and see a continuous, readable stream of logs showing training progress, learning curves, SPS, and reward decompositions.

Implement or update Daddy/train_big.py and any minimal supporting changes required in Daddy/ so that this works end-to-end.
