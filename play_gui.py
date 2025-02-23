import pickle
from typing import Dict

# pylint: disable=no-member
import pygame

from snake_env import ImprovedSnakeEnv, State
from utils import SnakeActions, get_best_action


def play_snake_gui(grid_size: int, block_size: int) -> None:
    with open("models/q_table.pkl", "rb") as f:
        Q_table: Dict[State, Dict[str, float]] = pickle.load(f)

    env = ImprovedSnakeEnv(
        grid_size=grid_size, block_size=block_size, render_mode="human"
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
    play_snake_gui(grid_size=15, block_size=40)
