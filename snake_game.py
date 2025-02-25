import argparse
import datetime
import os
import sys
import time
from typing import Optional

# pylint: disable=no-member
import pygame

from constants import (
    FilePaths,
    ModelType,
    SnakeActions,
    SnakeConfig,
    TrainingConfig,
    create_default_environment,
)


def train_agent(model_type: str, num_episodes: int, suffix: Optional[str]) -> None:
    env = create_default_environment(SnakeConfig.RENDER_MODE_NONE)
    try:
        agent = ModelType.create_agent(model_type)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    model_name = agent.train(env, num_episodes, suffix)

    os.makedirs(FilePaths.MODELS_DIR, exist_ok=True)
    full_path = os.path.join(FilePaths.MODELS_DIR, f"{model_name}.pkl")
    agent.save(full_path)
    print(f"Model saved to {full_path}")


def handle_player_input() -> Optional[str]:
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


def wait_for_restart_or_quit() -> bool:
    waiting_for_input = True
    while waiting_for_input:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                return True
    return False


def play_agent(model_path: Optional[str]) -> None:
    env = create_default_environment(SnakeConfig.RENDER_MODE_HUMAN)
    use_model = False
    agent = None

    if model_path:
        use_model = True
        try:
            model_type = ModelType.extract_from_filename(os.path.basename(model_path))
            agent = ModelType.create_agent(model_type)
            agent.load(model_path)
        except ValueError as e:
            print(str(e))
            sys.exit(1)

    running = True
    while running:
        state = env.reset()
        step_count = 0
        current_action = env.random.choice(SnakeActions.all())
        game_over = False

        while not game_over and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if use_model and agent:
                action = agent.choose_action(state)
            else:
                new_action = handle_player_input()
                if new_action is not None and new_action != current_action:
                    current_action = new_action
                action = current_action

            next_state, _, done = env.step(action)
            state = next_state
            if done:
                game_over = True
                if use_model:
                    running = False

            env.render(step_count, str(step_count), "Game Over!" if game_over else "")

            if game_over and not use_model:
                running = wait_for_restart_or_quit()
                if running:
                    break

            step_count += 1

    env.close()
    total_time = (
        int(env.end_time - env.start_time)
        if env.end_time
        else int(time.time() - env.start_time)
    )
    print(
        f"Game Over! Final Score: {env.score}, Model Score: {env.model_score}, Time: {total_time}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--type", type=str, required=True)
    train_parser.add_argument(
        "--episodes", type=int, default=TrainingConfig.NUM_EPISODES
    )
    train_parser.add_argument("--suffix", type=str, default="")

    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--model", type=str, default="")

    args = parser.parse_args()

    if args.command == "train":
        train_agent(args.type, args.episodes, args.suffix)
    elif args.command == "play":
        model_path = args.model if args.model != "" else None
        play_agent(model_path)


if __name__ == "__main__":
    main()
