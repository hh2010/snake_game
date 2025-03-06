import argparse
import datetime
import os
import sys
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


def train_agent(
    model_type: str, num_episodes: int, suffix: Optional[str], enable_logging: bool
) -> None:
    env = create_default_environment(SnakeConfig.RENDER_MODE_NONE)
    try:
        agent = ModelType.create_agent(model_type, enable_logging)
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
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                return True


def play_agent(model_path: Optional[str], headless: bool, enable_logging: bool) -> None:
    env = create_default_environment(
        SnakeConfig.RENDER_MODE_NONE if headless else SnakeConfig.RENDER_MODE_HUMAN
    )
    use_model = False
    agent = None
    model_type = "human"

    if model_path:
        use_model = True
        try:
            try:
                model_type = ModelType.from_string(model_path)
                agent = ModelType.create_agent(model_type, enable_logging)
            except ValueError:
                model_type = ModelType.extract_from_filename(
                    os.path.basename(model_path)
                )
                agent = ModelType.create_agent(model_type, enable_logging)
                if os.path.exists(model_path):
                    agent.load(model_path)
        except ValueError as e:
            print(str(e))
            sys.exit(1)

    running = True
    state = env.reset()
    step_count = 0

    while running:
        if use_model and agent:
            action = agent.choose_action(state)
        else:
            if headless:
                running = False
                break
            new_action = handle_player_input()
            if new_action is not None:
                action = new_action
            else:
                action = env.direction

        next_state, _, done = env.step(action)
        state = next_state

        if not headless:
            env.render(step_count, str(step_count), "Game Over!" if done else "")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        if done:
            if use_model:
                running = False
            elif not headless:
                running = wait_for_restart_or_quit()
                if running:
                    state = env.reset()
                    step_count = 0
                    continue

        step_count += 1

    env.close()

    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print("=" * 50)
    print(f"Model Type: {model_type if use_model else 'human'}")
    print(f"Final Score: {env.score}")
    print(f"Model Score: {env.model_score}")
    print(f"Steps: {step_count}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--type", type=str, required=True)
    train_parser.add_argument(
        "--episodes", type=int, default=TrainingConfig.NUM_EPISODES
    )
    train_parser.add_argument("--suffix", type=str, default="")
    train_parser.add_argument(
        "--logging", action="store_true", help="Enable detailed logging"
    )

    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--model", type=str, default="")
    play_parser.add_argument("--headless", action="store_true", help="Run without GUI")
    play_parser.add_argument(
        "--logging", action="store_true", help="Enable detailed logging"
    )

    args = parser.parse_args()

    if args.command == "train":
        train_agent(args.type, args.episodes, args.suffix, args.logging)
    elif args.command == "play":
        model_path = args.model if args.model != "" else None
        play_agent(model_path, args.headless, args.logging)


if __name__ == "__main__":
    main()
