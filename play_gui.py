import pickle
import random
from typing import Any, Callable, Dict, List

# pylint: disable=no-member
import pygame

from snake_env import Action, ImprovedSnakeEnv, State


def dict_get(d: Dict[str, float]) -> Callable[[str], float]:
    return lambda k: d[k]


def get_best_action(
    Q_table: Dict[State, Dict[str, float]], state: State, actions: List[str]
) -> Action:
    if state not in Q_table:
        return random.choice(actions)
    return max(Q_table[state], key=dict_get(Q_table[state]))


def play_snake_gui() -> None:
    with open("models/q_table.pkl", "rb") as f:
        Q_table: Dict[State, Dict[str, float]] = pickle.load(f)

    env = ImprovedSnakeEnv(grid_size=15, block_size=40, render_mode="human")
    state = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = get_best_action(Q_table, state, ["UP", "DOWN", "LEFT", "RIGHT"])
        next_state, _, done = env.step(action)
        state = next_state
        env.render()

        if done:
            running = False

    env.close()
    print("Game Over!")


if __name__ == "__main__":
    play_snake_gui()
