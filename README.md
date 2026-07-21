# Throw Q-learning

Simple Streamlit app for training a 2-link planar arm to throw a ball toward a target with Approximate Q-learning.

## Setup

```bash
cd /Users/moriyasu/dev/throw-ql-workspace/throw-ql
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Web App

```bash
streamlit run app.py
```

The sidebar mode selector switches between distance throwing and wall rally. Distance throwing controls epsilon, learning rate, discount factor, episode count, target distance, random seed, an optional demo form hint, and motion playback speed. The target range is limited to 1.0-5.0 m, which is the reachable range for the current simplified arm and velocity constraints. Wall rally controls target catches from 1 to 10 and learns to move the arm tip into the returning ball after one ground bounce. The wall slowly moves between bounds, so repeated catches are intentionally harder than the fixed-wall version. You can train, reset, run a greedy evaluation, draw motion without learning, and save or load mode-specific weights.

## Run Tests

```bash
pytest
```

## Design Notes

- `environment.py` contains the 2D arm kinematics, discrete actions, simple projectile calculation, rewards, and normalized features.
- `agent.py` contains feature-based Approximate Q-learning with epsilon-greedy action selection and JSON weight persistence.
- `training.py` contains episode loops for training, scripted warm-start demos, motion-primitive plan learning, and greedy motion recording.
- `wall_task.py` contains the wall-rally environment, SARSA-style approximate updates, catch primitives, and wall motion recording.
- `app.py` contains only the Streamlit controls and visualization.

The feature vector includes action-conditioned terms such as `release:tip_vx`, `release:predicted_landing_delta`, and `shoulder+ elbow-:target_distance`. Training now uses a small set of pull/coast/brake/release motion-primitive plans and learns an approximate Q-value over plan features such as predicted landing delta, release step, impulse, and target distance. This keeps the task close to the crawler example: small action space, dense feedback, and clear state features. `Draw greedy motion` and `Draw demo motion` report their action sequences without changing weights.

Wall rally uses state features for arm angles/velocities, ball position/velocity, moving wall position/velocity, relative ball-to-tip distance, relative ball-to-wall distance, bounce state, catch window, rally progress, and action-conditioned versions of those features. Actions are catch primitives: `track`, `track-high`, and `track-low`. Rewards include dense distance-improvement shaping, catch bonuses, completion bonuses, and miss penalties.

See `DEVELOPMENT_NOTES.md` for a short running log of tuning changes.
