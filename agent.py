from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

from environment import Action, ArmState


FeatureFn = Callable[[ArmState, Action], Dict[str, float]]
ActionFn = Callable[[ArmState], Iterable[Action]]


class ApproximateQAgent:
    """Feature-based Q-learning agent."""

    def __init__(
        self,
        actions_fn: ActionFn,
        features_fn: FeatureFn,
        epsilon: float = 0.2,
        learning_rate: float = 0.1,
        discount: float = 0.9,
        seed: Optional[int] = None,
    ) -> None:
        self.actions_fn = actions_fn
        self.features_fn = features_fn
        self.epsilon = float(epsilon)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.weights: Dict[str, float] = {}
        self.rng = random.Random(seed)

    def q_value(self, state: ArmState, action: Action) -> float:
        return sum(self.weights.get(name, 0.0) * value for name, value in self.features_fn(state, action).items())

    def value(self, state: ArmState) -> float:
        actions = list(self.actions_fn(state))
        if not actions:
            return 0.0
        return max(self.q_value(state, action) for action in actions)

    def best_action(self, state: ArmState) -> Optional[Action]:
        actions = list(self.actions_fn(state))
        if not actions:
            return None
        scored = [(self.q_value(state, action), action) for action in actions]
        best_score = max(score for score, _ in scored)
        best_actions = [action for score, action in scored if score == best_score]
        return self.rng.choice(best_actions)

    def choose_action(self, state: ArmState, greedy: bool = False) -> Optional[Action]:
        actions = list(self.actions_fn(state))
        if not actions:
            return None
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.choice(actions)
        return self.best_action(state)

    def update(self, state: ArmState, action: Action, next_state: ArmState, reward: float) -> float:
        prediction = self.q_value(state, action)
        target = reward + self.discount * self.value(next_state)
        td_error = target - prediction
        for name, value in self.features_fn(state, action).items():
            self.weights[name] = self.weights.get(name, 0.0) + self.learning_rate * td_error * value
        return td_error

    def save_weights(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.weights, indent=2, sort_keys=True), encoding="utf-8")

    def load_weights(self, path: str | Path) -> None:
        self.weights = {key: float(value) for key, value in json.loads(Path(path).read_text(encoding="utf-8")).items()}

    def set_params(self, epsilon: float, learning_rate: float, discount: float) -> None:
        self.epsilon = float(epsilon)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
