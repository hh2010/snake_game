import pickle
from typing import Dict

from constants import FilePaths, SnakeActions, SnakeConfig
from snake_env import ImprovedSnakeEnv, State
from utils import get_best_action


def play_snake(render_mode: str) -> None:
    with open(FilePaths.Q_TABLE_PATH, "rb") as f:
        Q_table: Dict[State, Dict[str, float]] = pickle.load(f)

    env = ImprovedSnakeEnv(
        grid_size=SnakeConfig.DEFAULT_GRID_SIZE,
        block_size=SnakeConfig.DEFAULT_BLOCK_SIZE,
        render_mode=render_mode,
    )
    state = env.reset()
    total_reward = 0

    while True:
        action = get_best_action(state, Q_table, SnakeActions.all())
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break

    print(f"Game Over! Final Score: {total_reward}")


if __name__ == "__main__":
    play_snake(render_mode=SnakeConfig.RENDER_MODE_NONE)
