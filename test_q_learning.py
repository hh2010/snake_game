import pickle
import random
from typing import Callable, Dict, List

from snake_env import Action, ImprovedSnakeEnv, State


def dict_get(d: Dict[str, float]) -> Callable[[str], float]:
    return lambda k: d[k]


def get_best_action(
    Q_table: Dict[State, Dict[str, float]], state: State, actions: List[str]
) -> Action:
    if state not in Q_table:
        return random.choice(actions)
    return max(Q_table[state], key=dict_get(Q_table[state]))


def play_snake(
    grid_size: int, block_size: int, render_mode: str, actions: List[str]
) -> None:
    with open("./models/q_table.pkl", "rb") as f:
        Q_table: Dict[State, Dict[str, float]] = pickle.load(f)

    env = ImprovedSnakeEnv(
        grid_size=grid_size, block_size=block_size, render_mode=render_mode
    )
    state = env.reset()
    total_reward = 0

    while True:
        action = get_best_action(Q_table, state, actions)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break

    print(f"Game Over! Final Score: {total_reward}")


if __name__ == "__main__":
    play_snake(
        grid_size=10,
        block_size=40,
        render_mode="none",
        actions=["UP", "DOWN", "LEFT", "RIGHT"],
    )
