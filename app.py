from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import streamlit as st

from agent import ApproximateQAgent
from environment import EpisodeTrace, ThrowingArmEnvironment
from training import (
    EpisodeStats,
    MotionFrame,
    evaluate_greedy,
    make_env_and_agent,
    record_demo_motion,
    record_greedy_motion,
    train_agent,
    warm_start_agent,
)
from wall_task import (
    WallEpisodeStats,
    WallMotionFrame,
    WallTrace,
    evaluate_wall_greedy,
    make_wall_env_and_agent,
    record_wall_motion,
    train_wall_agent,
    wall_success_rate,
)


WEIGHTS_PATH = Path("weights.json")
WALL_WEIGHTS_PATH = Path("wall_weights.json")


def ensure_state() -> None:
    if "env" not in st.session_state or "agent" not in st.session_state:
        env, agent = make_env_and_agent(
            target_distance=3.0,
            epsilon=0.35,
            learning_rate=0.15,
            discount=0.9,
            seed=0,
        )
        st.session_state.env = env
        st.session_state.agent = agent
        st.session_state.history = []
        st.session_state.trace = env.episode_trace(0.0)
        st.session_state.motion_frames = []
        st.session_state.motion_actions = []
        st.session_state.motion_label = ""
        st.session_state.last_message = "Ready"


def reset_agent(target_distance: float, epsilon: float, learning_rate: float, discount: float, seed: int) -> None:
    env, agent = make_env_and_agent(target_distance, epsilon, learning_rate, discount, seed)
    st.session_state.env = env
    st.session_state.agent = agent
    st.session_state.history = []
    st.session_state.trace = env.episode_trace(0.0)
    st.session_state.motion_frames = []
    st.session_state.motion_actions = []
    st.session_state.motion_label = ""
    st.session_state.last_message = "Reset complete"


def ensure_wall_state() -> None:
    if "wall_env" not in st.session_state or "wall_agent" not in st.session_state:
        env, agent = make_wall_env_and_agent(
            target_catches=3,
            epsilon=0.25,
            learning_rate=0.04,
            discount=0.95,
            seed=0,
        )
        st.session_state.wall_env = env
        st.session_state.wall_agent = agent
        st.session_state.wall_history = []
        st.session_state.wall_trace = env.episode_trace(0.0)
        st.session_state.wall_motion_frames = []
        st.session_state.wall_motion_actions = []
        st.session_state.wall_last_message = "Wall rally ready"


def reset_wall_agent(target_catches: int, epsilon: float, learning_rate: float, discount: float, seed: int) -> None:
    env, agent = make_wall_env_and_agent(target_catches, epsilon, learning_rate, discount, seed)
    st.session_state.wall_env = env
    st.session_state.wall_agent = agent
    st.session_state.wall_history = []
    st.session_state.wall_trace = env.episode_trace(0.0)
    st.session_state.wall_motion_frames = []
    st.session_state.wall_motion_actions = []
    st.session_state.wall_last_message = "Wall rally reset complete"


def sync_params(target_distance: float, epsilon: float, learning_rate: float, discount: float) -> None:
    env: ThrowingArmEnvironment = st.session_state.env
    agent: ApproximateQAgent = st.session_state.agent
    env.target_distance = target_distance
    agent.set_params(epsilon, learning_rate, discount)


def sync_wall_params(target_catches: int, epsilon: float, learning_rate: float, discount: float) -> None:
    env = st.session_state.wall_env
    agent: ApproximateQAgent = st.session_state.wall_agent
    env.target_catches = int(target_catches)
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


def draw_wall_frame(frame: WallMotionFrame, trace: WallTrace) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, color="#333333", linewidth=1)
    ax.axvline(frame.wall_x, color="#555555", linewidth=3, label="wall")
    ax.axvspan(0.85, 1.55, color="#2ca02c", alpha=0.08, label="catch zone")

    base, elbow, tip = frame.joints
    ax.plot([base[0], elbow[0], tip[0]], [base[1], elbow[1], tip[1]], "-o", color="#1f77b4", linewidth=4)
    if trace.ball_path:
        ax.plot([p[0] for p in trace.ball_path], [p[1] for p in trace.ball_path], color="#ff7f0e", alpha=0.35)
    if trace.catch_points:
        ax.scatter([p[0] for p in trace.catch_points], [p[1] for p in trace.catch_points], color="#2ca02c", s=70)
    ax.scatter([frame.ball_position[0]], [frame.ball_position[1]], color="#ff7f0e", s=70, label="ball")

    ax.set_title(f"step {frame.step} catches {frame.catches}/{trace.target_catches}")
    ax.set_xlim(-0.4, 4.2)
    ax.set_ylim(-0.1, 2.4)
    ax.set_xlabel("distance")
    ax.set_ylabel("height")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def draw_wall_trace(trace: WallTrace) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, color="#333333", linewidth=1)
    wall_x = trace.wall_path[-1][0] if trace.wall_path else 3.6
    ax.axvline(wall_x, color="#555555", linewidth=3, label="wall")
    if trace.wall_path:
        ax.plot([p[0] for p in trace.wall_path], [2.25 for _ in trace.wall_path], color="#555555", alpha=0.35)
    ax.axvspan(0.85, 1.55, color="#2ca02c", alpha=0.08, label="catch zone")
    if trace.joint_positions:
        base, elbow, tip = trace.joint_positions[-1]
        ax.plot([base[0], elbow[0], tip[0]], [base[1], elbow[1], tip[1]], "-o", color="#1f77b4", linewidth=4)
    if trace.ball_path:
        ax.plot([p[0] for p in trace.ball_path], [p[1] for p in trace.ball_path], color="#ff7f0e", label="ball path")
        ax.scatter([trace.ball_path[-1][0]], [trace.ball_path[-1][1]], color="#ff7f0e", s=70)
    if trace.catch_points:
        ax.scatter([p[0] for p in trace.catch_points], [p[1] for p in trace.catch_points], color="#2ca02c", s=70)
    ax.set_xlim(-0.4, 4.2)
    ax.set_ylim(-0.1, 2.4)
    ax.set_xlabel("distance")
    ax.set_ylabel("height")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def draw_wall_learning_curve(history: list[WallEpisodeStats]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    if history:
        ax.plot([item.episode for item in history], [item.catches for item in history], color="#1f77b4")
    ax.set_xlabel("episode")
    ax.set_ylabel("catches")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def success_rate(history: list[EpisodeStats]) -> float:
    if not history:
        return 0.0
    window = history[-50:]
    return sum(1 for item in window if item.success) / len(window)


st.set_page_config(page_title="Throw Q-learning", layout="wide")
ensure_state()
st.title("Throw Q-learning")

mode = st.sidebar.radio("mode", ["Throw distance", "Wall rally"], horizontal=True)

if mode == "Wall rally":
    ensure_wall_state()
    with st.sidebar:
        wall_epsilon = st.slider("epsilon", 0.0, 1.0, 0.25, 0.01, key="wall_epsilon")
        wall_learning_rate = st.slider("learning rate", 0.0, 1.0, 0.04, 0.01, key="wall_learning_rate")
        wall_discount = st.slider("discount factor", 0.0, 0.99, 0.95, 0.01, key="wall_discount")
        wall_episodes = st.number_input("episode count", min_value=1, max_value=5000, value=200, step=50, key="wall_episodes")
        target_catches = st.slider("target catches", 1, 10, 3, 1)
        wall_seed = st.number_input("random seed", min_value=0, max_value=999999, value=0, step=1, key="wall_seed")
        wall_motion_delay = st.slider("motion delay", 0.01, 0.3, 0.03, 0.01, key="wall_motion_delay")

        sync_wall_params(target_catches, wall_epsilon, wall_learning_rate, wall_discount)

        if st.button("Start wall training", use_container_width=True):
            history, trace = train_wall_agent(st.session_state.wall_env, st.session_state.wall_agent, int(wall_episodes))
            offset = len(st.session_state.wall_history)
            for index, item in enumerate(history, start=1):
                item.episode = offset + index
            st.session_state.wall_history.extend(history)
            st.session_state.wall_trace = trace
            st.session_state.wall_motion_actions = []
            st.session_state.wall_last_message = f"Trained wall rally {wall_episodes} episodes"

        if st.button("Greedy wall evaluation", use_container_width=True):
            stats, trace = evaluate_wall_greedy(st.session_state.wall_env, st.session_state.wall_agent, episodes=1)
            st.session_state.wall_trace = trace
            st.session_state.wall_motion_actions = []
            st.session_state.wall_last_message = f"Greedy catches {stats[-1].catches}/{stats[-1].target_catches}"

        draw_wall = st.button("Draw wall motion", use_container_width=True, type="primary")

        if st.button("Save wall weights", use_container_width=True):
            st.session_state.wall_agent.save_weights(WALL_WEIGHTS_PATH)
            st.session_state.wall_last_message = f"Saved {WALL_WEIGHTS_PATH}"

        if st.button("Load wall weights", use_container_width=True):
            if WALL_WEIGHTS_PATH.exists():
                st.session_state.wall_agent.load_weights(WALL_WEIGHTS_PATH)
                st.session_state.wall_last_message = f"Loaded {WALL_WEIGHTS_PATH}"
            else:
                st.session_state.wall_last_message = f"{WALL_WEIGHTS_PATH} not found"

        if st.button("Reset wall mode", use_container_width=True):
            reset_wall_agent(target_catches, wall_epsilon, wall_learning_rate, wall_discount, int(wall_seed))

    wall_history: list[WallEpisodeStats] = st.session_state.wall_history
    wall_trace: WallTrace = st.session_state.wall_trace
    wall_motion = None
    if draw_wall:
        weights_before = dict(st.session_state.wall_agent.weights)
        wall_motion = record_wall_motion(st.session_state.wall_env, st.session_state.wall_agent)
        st.session_state.wall_trace = wall_motion.trace
        st.session_state.wall_motion_frames = wall_motion.frames
        st.session_state.wall_motion_actions = wall_motion.actions
        st.session_state.wall_last_message = (
            f"Rendered wall motion catches {wall_motion.stats.catches}/{wall_motion.stats.target_catches}"
        )
        if st.session_state.wall_agent.weights != weights_before:
            st.error("Weights changed during wall motion rendering.")
        wall_trace = st.session_state.wall_trace

    latest_wall = wall_history[-1] if wall_history else None
    wall_cols = st.columns(5)
    wall_cols[0].metric("episode", latest_wall.episode if latest_wall else 0)
    wall_cols[1].metric("episode reward", f"{latest_wall.total_reward:.2f}" if latest_wall else "0.00")
    wall_cols[2].metric("catches", f"{wall_trace.catches}/{wall_trace.target_catches}")
    wall_cols[3].metric("success rate", f"{wall_success_rate(wall_history) * 100:.1f}%")
    wall_cols[4].metric("weights", len(st.session_state.wall_agent.weights))
    st.caption(st.session_state.wall_last_message)

    left, right = st.columns([1.2, 1.0])
    with left:
        wall_slot = st.empty()
        if wall_motion is not None:
            for frame in wall_motion.frames:
                fig = draw_wall_frame(frame, wall_motion.trace)
                wall_slot.pyplot(fig)
                plt.close(fig)
                time.sleep(wall_motion_delay)
        else:
            fig = draw_wall_trace(wall_trace)
            wall_slot.pyplot(fig)
            plt.close(fig)
    with right:
        fig = draw_wall_learning_curve(wall_history)
        st.pyplot(fig)
        plt.close(fig)
        if st.session_state.wall_motion_actions:
            st.subheader("Wall motion sequence")
            st.write(
                {
                    "catches": wall_trace.catches,
                    "target catches": wall_trace.target_catches,
                    "success": wall_trace.success,
                    "missed": wall_trace.missed,
                    "motion reward": round(wall_trace.total_reward, 3),
                }
            )
            st.dataframe(
                [{"step": index + 1, "action": action} for index, action in enumerate(st.session_state.wall_motion_actions)],
                use_container_width=True,
                hide_index=True,
            )
        if st.session_state.wall_agent.weights:
            st.dataframe(
                sorted(st.session_state.wall_agent.weights.items()),
                use_container_width=True,
                hide_index=True,
                column_config={"0": "feature", "1": "weight"},
            )
    st.stop()

with st.sidebar:
    epsilon = st.slider("epsilon", 0.0, 1.0, 0.35, 0.01)
    learning_rate = st.slider("learning rate", 0.0, 1.0, 0.15, 0.01)
    discount = st.slider("discount factor", 0.0, 0.99, 0.9, 0.01)
    episodes = st.number_input("episode count", min_value=1, max_value=5000, value=50, step=50)
    target_distance = st.slider("target distance", 1.0, 5.0, 3.0, 0.1)
    random_seed = st.number_input("random seed", min_value=0, max_value=999999, value=0, step=1)
    use_warm_start = st.checkbox("use demo form hint", value=True)
    demo_episodes = st.number_input("demo episodes", min_value=1, max_value=500, value=30, step=5)
    motion_delay = st.slider("motion delay", 0.02, 0.5, 0.08, 0.01)
    show_trail = st.checkbox("show motion trail", value=True)

    sync_params(target_distance, epsilon, learning_rate, discount)

    if st.button("Start training", use_container_width=True):
        warm_history = []
        if use_warm_start and not st.session_state.history:
            warm_history, trace = warm_start_agent(
                st.session_state.env,
                st.session_state.agent,
                int(demo_episodes),
            )
            for index, item in enumerate(warm_history, start=1):
                item.episode = index
            st.session_state.history.extend(warm_history)
        history, trace = train_agent(st.session_state.env, st.session_state.agent, int(episodes))
        offset = len(st.session_state.history)
        for index, item in enumerate(history, start=1):
            item.episode = offset + index
        st.session_state.history.extend(history)
        st.session_state.trace = trace
        st.session_state.motion_actions = []
        st.session_state.motion_label = ""
        if warm_history:
            st.session_state.last_message = f"Used {demo_episodes} demo-form episodes, then trained {episodes} RL episodes"
        else:
            st.session_state.last_message = f"Trained {episodes} episodes"

    if st.button("Greedy evaluation", use_container_width=True):
        stats, trace = evaluate_greedy(st.session_state.env, st.session_state.agent, episodes=1)
        st.session_state.trace = trace
        st.session_state.motion_actions = []
        st.session_state.motion_label = ""
        st.session_state.last_message = f"Greedy reward {stats[-1].total_reward:.2f}"

    with st.container(border=True):
        st.markdown("Motion playback")
        motion_cols = st.columns(2)
        draw_demo_motion = motion_cols[0].button("Demo", use_container_width=True, type="secondary")
        draw_motion = motion_cols[1].button("Greedy", use_container_width=True, type="primary")

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
    st.session_state.motion_actions = motion_to_render.actions
    st.session_state.motion_label = "Greedy motion sequence"
    st.session_state.last_message = f"Rendered greedy motion reward {motion_to_render.stats.total_reward:.2f}"
    if st.session_state.agent.weights != weights_before:
        st.error("Weights changed during motion rendering.")
    trace = st.session_state.trace
elif draw_demo_motion:
    weights_before = dict(st.session_state.agent.weights)
    motion_to_render = record_demo_motion(st.session_state.env, st.session_state.agent)
    st.session_state.trace = motion_to_render.trace
    st.session_state.motion_frames = motion_to_render.frames
    st.session_state.motion_actions = motion_to_render.actions
    st.session_state.motion_label = "Demo motion sequence"
    st.session_state.last_message = f"Rendered demo motion reward {motion_to_render.stats.total_reward:.2f}"
    if st.session_state.agent.weights != weights_before:
        st.error("Weights changed during demo rendering.")
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
    if st.session_state.motion_actions:
        st.subheader(st.session_state.motion_label or "Motion sequence")
        release_step = next(
            (index + 1 for index, action in enumerate(st.session_state.motion_actions) if "release" in action),
            None,
        )
        landing_x = trace.landing_point[0] if trace.landing_point is not None else None
        st.write(
            {
                "release step": release_step,
                "landing x": round(landing_x, 3) if landing_x is not None else None,
                "landing error": round(trace.landing_error, 3) if trace.landing_error is not None else None,
                "motion reward": round(trace.total_reward, 3),
            }
        )
        st.dataframe(
            [{"step": index + 1, "action": action} for index, action in enumerate(st.session_state.motion_actions)],
            use_container_width=True,
            hide_index=True,
        )
    if st.session_state.agent.weights:
        st.dataframe(
            sorted(st.session_state.agent.weights.items()),
            use_container_width=True,
            hide_index=True,
            column_config={"0": "feature", "1": "weight"},
        )
