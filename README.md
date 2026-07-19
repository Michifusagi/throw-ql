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

The sidebar controls epsilon, learning rate, discount factor, episode count, target distance, random seed, optional warm-start demo training, and motion playback speed. You can train, reset, run a greedy evaluation, use the Motion playback buttons to draw one greedy motion or the fixed demo motion without learning, and save or load `weights.json`.

## Run Tests

```bash
pytest
```

## Design Notes

- `environment.py` contains the 2D arm kinematics, discrete actions, simple projectile calculation, rewards, and normalized features.
- `agent.py` contains feature-based Approximate Q-learning with epsilon-greedy action selection and JSON weight persistence.
- `training.py` contains episode loops for training, scripted warm-start demos, and greedy motion recording.
- `app.py` contains only the Streamlit controls and visualization.

The feature vector includes action-conditioned terms such as `release:tip_vx` and `shoulder+ elbow-:relative_target`, so the agent can learn different meanings for the same state under different actions. `Draw greedy motion` and `Draw demo motion` report their action sequences without changing weights.

See `DEVELOPMENT_NOTES.md` for a short running log of tuning changes.
