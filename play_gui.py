import argparse
import pickle
import time
from typing import Dict, Optional

# pylint: disable=no-member
import pygame

from constants import FilePaths, RandomState, SnakeActions, SnakeConfig
from snake_env import ImprovedSnakeEnv, State
from utils import get_best_action


def get_keyboard_action() -> Optional[str]:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        return SnakeActions.UP
    elif keys[pygame.K_DOWN]:
        return SnakeActions.DOWN
    elif keys[pygame.K_LEFT]:
        return SnakeActions.LEFT
    elif keys[pygame.K_RIGHT]:
        return SnakeActions.RIGHT
    return None


def play_snake_gui(use_model: bool) -> None:
    Q_table: Optional[Dict[State, Dict[str, float]]] = None
    if use_model:
        with open(FilePaths.Q_TABLE_PATH, "rb") as f:
            Q_table = pickle.load(f)

    env = ImprovedSnakeEnv(
        grid_size=SnakeConfig.DEFAULT_GRID_SIZE,
        block_size=SnakeConfig.DEFAULT_BLOCK_SIZE,
        render_mode=SnakeConfig.RENDER_MODE_HUMAN,
    )
    running = True

    while running:
        # Reset the random state to ensure reproducibility
        RandomState.RANDOM.seed(RandomState.SEED)

        state = env.reset()
        step_count = 0
        current_action = RandomState.RANDOM.choice(SnakeActions.all())
        game_over = False

        while not game_over and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if use_model and Q_table is not None:
                action = get_best_action(state, Q_table, SnakeActions.all())
            else:
                new_action = get_keyboard_action()
                if new_action is not None and new_action != current_action:
                    current_action = new_action
                action = current_action

            next_state, _, done = env.step(action)
            state = next_state

            if done:
                game_over = True
                if use_model:
                    running = False

            env.render(
                step_count=step_count,
                step_text=str(step_count),
                game_over_text="Game Over!" if game_over else "",
            )

            if game_over and not use_model:
                waiting_for_input = True
                while waiting_for_input and running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            waiting_for_input = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_RETURN:
                                waiting_for_input = False
                                game_over = False
                                break
                    env.render(
                        step_count=step_count,
                        step_text=str(step_count),
                        game_over_text="Game Over!",
                    )
                if not game_over:
                    break

    env.close()
    print(
        f"Game Over! Final Score: {env.score}, Model Score: {env.model_score}, Time: {int(env.end_time - env.start_time if env.end_time else time.time() - env.start_time)}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play Snake with AI or keyboard controls"
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Use keyboard controls instead of the trained model",
    )
    args = parser.parse_args()

    play_snake_gui(use_model=not args.no_model)
