# Slim Pokémon Red Agent (Daddy/)

Implementation of the hierarchical CNN→GRU→LSTM→SSM DQN agent described in `DESIGN_SLIM_MODEL_V2.md`.

## Layout
- `networks.py` – hierarchical encoder producing Q-values + recurrent state
- `agent.py` – SlimHierarchicalDQN wrapper (RND, novelty, Bayes monitor, replay)
- `train_slim.py` – multi-env training loop with replay + target network
- `debug_rollout.py` – quick rollout script for integration testing
- `rnd.py`, `flags.py`, `bayes_quests.py`, `curriculum.py`, `replay_buffer.py`, `logging_utils.py`, `video_utils.py`

## Quickstart
```
python -m Daddy.train_slim --total-steps 20000 --num-envs 4 --headless
python -m Daddy.debug_rollout --steps 500 --video gif
```

See the design docs in this directory plus `docs/` for the full contract.
