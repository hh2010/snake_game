import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, List, Tuple, TypeAlias

Point: TypeAlias = tuple[int, int]
State: TypeAlias = tuple[int, int, int, int, int, int, int, int]
Action: TypeAlias = str


@dataclass(frozen=True)
class SnakeActions:
    UP: Final[str] = "UP"
    DOWN: Final[str] = "DOWN"
    LEFT: Final[str] = "LEFT"
    RIGHT: Final[str] = "RIGHT"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT]


@dataclass(frozen=True)
class SnakeConfig:
    DEFAULT_GRID_SIZE: Final[int] = 15
    DEFAULT_BLOCK_SIZE: Final[int] = 40
    RENDER_MODE_HUMAN: Final[str] = "human"
    RENDER_MODE_NONE: Final[str] = "none"
    GAME_SPEED: Final[int] = 10
    FONT_SIZE: Final[int] = 36
    METRICS_FONT_SIZE: Final[int] = 24


@dataclass(frozen=True)
class Colors:
    BLACK: Final[tuple[int, int, int]] = (0, 0, 0)
    GREEN: Final[tuple[int, int, int]] = (0, 255, 0)
    RED: Final[tuple[int, int, int]] = (255, 0, 0)
    WHITE: Final[tuple[int, int, int]] = (255, 255, 255)


@dataclass(frozen=True)
class TrainingConfig:
    ALPHA: Final[float] = 0.1
    GAMMA: Final[float] = 0.9
    EPSILON_START: Final[float] = 1.0
    EPSILON_DECAY: Final[float] = 0.995
    EPSILON_MIN: Final[float] = 0.01
    NUM_EPISODES: Final[int] = 4990
    PROGRESS_REPORT_INTERVAL: Final[int] = 100


@dataclass(frozen=True)
class RewardConfig:
    COLLISION_PENALTY: Final[int] = -100
    FOOD_REWARD: Final[int] = 50
    CLOSER_TO_FOOD: Final[int] = 1
    AWAY_FROM_FOOD: Final[int] = -1


@dataclass(frozen=True)
class FilePaths:
    MODELS_DIR: Final[Path] = Path("models")
    OUTPUTS_DIR: Final[Path] = Path("outputs")
    Q_TABLE_PATH: Final[Path] = MODELS_DIR / "q_table.pkl"
    TRAINING_PLOT_PATH: Final[Path] = OUTPUTS_DIR / "training_plot.png"
    TRAINING_REWARDS_PATH: Final[Path] = OUTPUTS_DIR / "training_rewards.csv"


@dataclass(frozen=True)
class RandomState:
    SEED: Final[int] = 42
    TRAINING_SEED: Final[int] = 69


@dataclass(frozen=True)
class PlotConfig:
    FIGURE_SIZE: Final[Tuple[int, int]] = (10, 6)
    TITLE: Final[str] = "Training Progress of Q-Learning Snake Agent"
    X_LABEL: Final[str] = "Episode"
    Y_LABEL: Final[str] = "Total Reward"


@dataclass(frozen=True)
class ModelType:
    QLEARNING: Final[str] = "qlearning"

    @staticmethod
    def from_string(model_type: str) -> str:
        model_type = model_type.lower()
        if model_type in [ModelType.QLEARNING]:
            return ModelType.QLEARNING
        raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def create_agent(model_type: str) -> Any:
        model_type = ModelType.from_string(model_type)
        if model_type == ModelType.QLEARNING:
            from agents.qlearning_agent import QLearningAgent

            return QLearningAgent()
        raise ValueError(f"No agent implementation for model type: {model_type}")

    @staticmethod
    def extract_from_filename(filename: str) -> str:
        # First check if the filename is just the model type itself
        if filename.lower() in [ModelType.QLEARNING]:
            return filename.lower()

        # Extract model type from filename like "20250224230142_qlearning.pkl"
        pattern = r"_(\w+)(?:_|\.)"
        match = re.search(pattern, filename)
        if match:
            return match.group(1)

        raise ValueError(f"Could not extract model type from filename: {filename}")


def create_default_environment(render_mode: str) -> Any:
    from snake_env import ImprovedSnakeEnv

    return ImprovedSnakeEnv(
        grid_size=SnakeConfig.DEFAULT_GRID_SIZE,
        block_size=SnakeConfig.DEFAULT_BLOCK_SIZE,
        render_mode=render_mode,
    )
