from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import streamlit as st

from agent import ApproximateQAgent
from environment import EpisodeTrace, ThrowingArmEnvironment
from training import EpisodeStats, MotionFrame, evaluate_greedy, make_env_and_agent, record_greedy_motion, train_agent


WEIGHTS_PATH = Path("weights.json")


def ensure_state() -> None:
    if "env" not in st.session_state or "agent" not in st.session_state:
        env, agent = make_env_and_agent(
            target_distance=5.0,
            epsilon=0.2,
            learning_rate=0.15,
            discount=0.9,
            seed=0,
        )
        st.session_state.env = env
        st.session_state.agent = agent
        st.session_state.history = []
        st.session_state.trace = env.episode_trace(0.0)
        st.session_state.motion_frames = []
        st.session_state.last_message = "Ready"


def reset_agent(target_distance: float, epsilon: float, learning_rate: float, discount: float, seed: int) -> None:
    env, agent = make_env_and_agent(target_distance, epsilon, learning_rate, discount, seed)
    st.session_state.env = env
    st.session_state.agent = agent
    st.session_state.history = []
    st.session_state.trace = env.episode_trace(0.0)
    st.session_state.motion_frames = []
    st.session_state.last_message = "Reset complete"


def sync_params(target_distance: float, epsilon: float, learning_rate: float, discount: float) -> None:
    env: ThrowingArmEnvironment = st.session_state.env
    agent: ApproximateQAgent = st.session_state.agent
    env.target_distance = target_distance
    agent.set_params(epsilon, learning_rate, discount)


def draw_trace(trace: EpisodeTrace, target_distance: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, color="#333333", linewidth=1)
    ax.axvline(target_distance, color="#2ca02c", linestyle="--", label="target")
    ax.scatter([target_distance], [0], color="#2ca02c", s=80)

    if trace.joint_positions:
        base, elbow, tip = trace.joint_positions[-1]
        ax.plot([base[0], elbow[0], tip[0]], [base[1], elbow[1], tip[1]], "-o", color="#1f77b4", linewidth=4)
    if trace.ball_path:
        xs = [point[0] for point in trace.ball_path]
        ys = [point[1] for point in trace.ball_path]
        ax.plot(xs, ys, color="#ff7f0e", label="ball path")
    if trace.landing_point is not None:
        ax.scatter([trace.landing_point[0]], [0], color="#d62728", s=70, label="landing")

    ax.set_xlim(-1.0, max(7.0, target_distance + 2.0))
    ax.set_ylim(-0.2, 3.0)
    ax.set_xlabel("distance")
    ax.set_ylabel("height")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def draw_motion_frame(
    frame: MotionFrame,
    trace: EpisodeTrace,
    target_distance: float,
    show_trail: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, color="#333333", linewidth=1)
    ax.axvline(target_distance, color="#2ca02c", linestyle="--", label="target")
    ax.scatter([target_distance], [0], color="#2ca02c", s=80)

    base, elbow, tip = frame.joints
    ax.plot([base[0], elbow[0], tip[0]], [base[1], elbow[1], tip[1]], "-o", color="#1f77b4", linewidth=4)

    if show_trail and trace.ball_path and frame.ball_position is not None:
        trail = [point for point in trace.ball_path if point[0] <= frame.ball_position[0]]
        if trail:
            ax.plot([point[0] for point in trail], [point[1] for point in trail], color="#ff7f0e", alpha=0.7)

    if frame.ball_position is not None:
        ax.scatter([frame.ball_position[0]], [frame.ball_position[1]], color="#ff7f0e", s=70, label="ball")
    if trace.landing_point is not None:
        ax.scatter([trace.landing_point[0]], [0], color="#d62728", s=70, label="landing")

    ax.set_title(f"step {frame.step}")
    ax.set_xlim(-1.0, max(7.0, target_distance + 2.0))
    ax.set_ylim(-0.2, 3.0)
    ax.set_xlabel("distance")
    ax.set_ylabel("height")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def draw_learning_curve(history: list[EpisodeStats]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    if history:
        episodes = [item.episode for item in history]
        rewards = [item.total_reward for item in history]
        ax.plot(episodes, rewards, color="#1f77b4")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def success_rate(history: list[EpisodeStats]) -> float:
    if not history:
        return 0.0
    window = history[-50:]
    return sum(1 for item in window if item.success) / len(window)


ensure_state()

st.set_page_config(page_title="Throw Q-learning", layout="wide")
st.title("Throw Q-learning")

with st.sidebar:
    epsilon = st.slider("epsilon", 0.0, 1.0, 0.2, 0.01)
    learning_rate = st.slider("learning rate", 0.0, 1.0, 0.15, 0.01)
    discount = st.slider("discount factor", 0.0, 0.99, 0.9, 0.01)
    episodes = st.number_input("episode count", min_value=1, max_value=5000, value=200, step=50)
    target_distance = st.slider("target distance", 1.0, 8.0, 5.0, 0.1)
    random_seed = st.number_input("random seed", min_value=0, max_value=999999, value=0, step=1)
    motion_delay = st.slider("motion delay", 0.02, 0.5, 0.08, 0.01)
    show_trail = st.checkbox("show motion trail", value=True)

    sync_params(target_distance, epsilon, learning_rate, discount)

    if st.button("Start training", use_container_width=True):
        history, trace = train_agent(st.session_state.env, st.session_state.agent, int(episodes))
        offset = len(st.session_state.history)
        for index, item in enumerate(history, start=1):
            item.episode = offset + index
        st.session_state.history.extend(history)
        st.session_state.trace = trace
        st.session_state.last_message = f"Trained {episodes} episodes"

    if st.button("Greedy evaluation", use_container_width=True):
        stats, trace = evaluate_greedy(st.session_state.env, st.session_state.agent, episodes=1)
        st.session_state.trace = trace
        st.session_state.last_message = f"Greedy reward {stats[-1].total_reward:.2f}"

    draw_motion = st.button("Draw greedy motion", use_container_width=True)

    if st.button("Save weights", use_container_width=True):
        st.session_state.agent.save_weights(WEIGHTS_PATH)
        st.session_state.last_message = f"Saved {WEIGHTS_PATH}"

    if st.button("Load weights", use_container_width=True):
        if WEIGHTS_PATH.exists():
            st.session_state.agent.load_weights(WEIGHTS_PATH)
            st.session_state.last_message = f"Loaded {WEIGHTS_PATH}"
        else:
            st.session_state.last_message = f"{WEIGHTS_PATH} not found"

    if st.button("Reset", use_container_width=True):
        reset_agent(target_distance, epsilon, learning_rate, discount, int(random_seed))

history: list[EpisodeStats] = st.session_state.history
trace: EpisodeTrace = st.session_state.trace

motion_to_render = None
if draw_motion:
    weights_before = dict(st.session_state.agent.weights)
    motion_to_render = record_greedy_motion(st.session_state.env, st.session_state.agent)
    st.session_state.trace = motion_to_render.trace
    st.session_state.motion_frames = motion_to_render.frames
    st.session_state.last_message = f"Rendered greedy motion reward {motion_to_render.stats.total_reward:.2f}"
    if st.session_state.agent.weights != weights_before:
        st.error("Weights changed during motion rendering.")
    trace = st.session_state.trace

latest = history[-1] if history else None
cols = st.columns(5)
cols[0].metric("episode", latest.episode if latest else 0)
cols[1].metric("episode reward", f"{latest.total_reward:.2f}" if latest else "0.00")
cols[2].metric("landing error", f"{trace.landing_error:.3f}" if trace.landing_error is not None else "-")
cols[3].metric("success rate", f"{success_rate(history) * 100:.1f}%")
cols[4].metric("weights", len(st.session_state.agent.weights))

st.caption(st.session_state.last_message)

left, right = st.columns([1.2, 1.0])
with left:
    motion_slot = st.empty()
    if motion_to_render is not None:
        for frame in motion_to_render.frames:
            fig = draw_motion_frame(frame, motion_to_render.trace, target_distance, show_trail)
            motion_slot.pyplot(fig)
            plt.close(fig)
            time.sleep(motion_delay)
    else:
        fig = draw_trace(trace, target_distance)
        motion_slot.pyplot(fig)
        plt.close(fig)
with right:
    curve_fig = draw_learning_curve(history)
    st.pyplot(curve_fig)
    plt.close(curve_fig)
    if st.session_state.agent.weights:
        st.dataframe(
            sorted(st.session_state.agent.weights.items()),
            use_container_width=True,
            hide_index=True,
            column_config={"0": "feature", "1": "weight"},
        )
