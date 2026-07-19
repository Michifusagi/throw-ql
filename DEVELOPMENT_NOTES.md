# Development Notes

Short notes for tuning the throwing-arm Approximate Q-learning app. Add new entries at the top or bottom, keeping each entry to a few lines.

## 2026-07-19

- Initial app split physics/environment, Approximate Q-learning agent, training loops, and Streamlit UI into separate modules.
- Added action-conditioned features so the same state can score differently for `release`, joint actions, and `coast`.
- Added a fixed demo throw and warm-start training to expose the agent to a useful multi-step throw before epsilon-greedy learning.
- Added `Draw greedy motion` and `Draw demo motion` so learned behavior and the scripted demo can be inspected as frame-by-frame motion.
- Changed `max_steps` timeout to force a release, producing landing/error feedback instead of ending with no landing point.
- Added demo policy pretraining plus small manual priors to make the initial greedy policy closer to the scripted throw.
- Moved the reset pose slightly backward (`shoulder=-20deg`, `elbow=70deg`) and retuned the demo so it starts closer to a throwing wind-up.
