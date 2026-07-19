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

The sidebar controls epsilon, learning rate, discount factor, episode count, target distance, and random seed. You can train, reset, run a greedy evaluation, and save or load `weights.json`.

## Run Tests

```bash
pytest
```

## Design Notes

- `environment.py` contains the 2D arm kinematics, discrete actions, simple projectile calculation, rewards, and normalized features.
- `agent.py` contains feature-based Approximate Q-learning with epsilon-greedy action selection and JSON weight persistence.
- `training.py` contains episode loops for training and greedy evaluation.
- `app.py` contains only the Streamlit controls and visualization.
