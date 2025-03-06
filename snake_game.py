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
    agent_type: str, num_episodes: int, suffix: Optional[str], enable_logging: bool
) -> None:
    # Check if agent requires training
    if not ModelType.requires_training(agent_type):
        print(
            f"Error: Agent type '{agent_type}' ({ModelType.get_display_name(agent_type)}) doesn't require training."
        )
        print("Deterministic algorithms like Hamiltonian and BFS don't need training.")
        sys.exit(1)

    env = create_default_environment(SnakeConfig.RENDER_MODE_NONE)
    try:
        agent = ModelType.create_agent(agent_type, enable_logging)
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


def play_agent(
    agent_type: Optional[str],
    model_file: Optional[str],
    headless: bool,
    enable_logging: bool,
    game_speed: Optional[int] = None,
) -> None:
    env = create_default_environment(
        SnakeConfig.RENDER_MODE_NONE if headless else SnakeConfig.RENDER_MODE_HUMAN
    )

    # Set custom game speed if provided
    if game_speed is not None:
        env.game_speed = game_speed

    use_model = False
    agent = None
    agent_display_name = "human"

    if agent_type:
        use_model = True
        try:
            # Validate agent type
            agent_type = ModelType.from_string(agent_type)

            # Check if we're trying to use a model file with a deterministic agent
            if model_file and not ModelType.requires_training(agent_type):
                print(
                    f"Warning: Agent type '{agent_type}' ({ModelType.get_display_name(agent_type)}) doesn't use model files."
                )
                print(f"The provided model file '{model_file}' will be ignored.")
                model_file = None

            # Create the agent
            agent = ModelType.create_agent(agent_type, enable_logging)
            agent_display_name = ModelType.get_display_name(agent_type)

            # Load model file if needed
            if model_file and ModelType.requires_training(agent_type):
                if os.path.exists(model_file):
                    agent.load(model_file)
                else:
                    print(f"Error: Model file '{model_file}' not found.")
                    sys.exit(1)

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
    print(f"Agent Type: {agent_display_name if use_model else 'human'}")
    print(f"Final Score: {env.score}")
    print(f"Model Score: {env.model_score}")
    print(f"Steps: {step_count}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--agent-type",
        type=str,
        required=True,
        help="Type of agent to train (e.g., qlearning)",
    )
    train_parser.add_argument(
        "--episodes", type=int, default=TrainingConfig.NUM_EPISODES
    )
    train_parser.add_argument("--suffix", type=str, default="")
    train_parser.add_argument(
        "--logging", action="store_true", help="Enable detailed logging"
    )

    play_parser = subparsers.add_parser("play")
    play_parser.add_argument(
        "--agent-type",
        type=str,
        default="",
        help="Type of agent to use (e.g., qlearning, hamiltonian, bfs, bfs_hamiltonian)",
    )
    play_parser.add_argument(
        "--model-file",
        type=str,
        default="",
        help="Path to model file (only needed for learning agents like qlearning)",
    )
    play_parser.add_argument("--headless", action="store_true", help="Run without GUI")
    play_parser.add_argument(
        "--logging", action="store_true", help="Enable detailed logging"
    )
    play_parser.add_argument(
        "--speed", type=int, help=f"Game speed (default: {SnakeConfig.GAME_SPEED})"
    )

    args = parser.parse_args()

    if args.command == "train":
        train_agent(args.agent_type, args.episodes, args.suffix, args.logging)
    elif args.command == "play":
        agent_type = args.agent_type if args.agent_type != "" else None
        model_file = args.model_file if args.model_file != "" else None

        play_agent(agent_type, model_file, args.headless, args.logging, args.speed)


if __name__ == "__main__":
    main()
