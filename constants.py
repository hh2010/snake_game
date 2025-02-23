from dataclasses import dataclass
from pathlib import Path
from typing import Final, List, TypeAlias

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
    NUM_EPISODES: Final[int] = 5000
    GRID_SIZE: Final[int] = 10
    BLOCK_SIZE: Final[int] = 40


@dataclass(frozen=True)
class RewardConfig:
    COLLISION_PENALTY: Final[int] = -10
    FOOD_REWARD: Final[int] = 10
    CLOSER_TO_FOOD: Final[int] = 1
    AWAY_FROM_FOOD: Final[int] = -1


@dataclass(frozen=True)
class FilePaths:
    MODELS_DIR: Final[Path] = Path("models")
    OUTPUTS_DIR: Final[Path] = Path("outputs")
    Q_TABLE_PATH: Final[Path] = MODELS_DIR / "q_table.pkl"
    TRAINING_PLOT_PATH: Final[Path] = OUTPUTS_DIR / "training_plot.png"
