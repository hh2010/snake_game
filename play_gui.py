import pickle
from typing import Dict

# pylint: disable=no-member
import pygame

from constants import FilePaths, SnakeActions, SnakeConfig
from snake_env import ImprovedSnakeEnv, State
from utils import get_best_action


def play_snake_gui() -> None:
    with open(FilePaths.Q_TABLE_PATH, "rb") as f:
        Q_table: Dict[State, Dict[str, float]] = pickle.load(f)

    env = ImprovedSnakeEnv(
        grid_size=SnakeConfig.DEFAULT_GRID_SIZE,
        block_size=SnakeConfig.DEFAULT_BLOCK_SIZE,
        render_mode=SnakeConfig.RENDER_MODE_HUMAN,
    )
    state = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = get_best_action(state, Q_table, SnakeActions.all())
        next_state, _, done = env.step(action)
        state = next_state
        env.render()

        if done:
            running = False

    env.close()
    print("Game Over!")


if __name__ == "__main__":
    play_snake_gui()
