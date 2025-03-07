import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, List, Tuple, TypeAlias

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
    DEFAULT_GRID_SIZE: Final[int] = 16
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
    COLLISION_PENALTY: Final[int] = -2
    FOOD_REWARD: Final[int] = 1
    CLOSER_TO_FOOD: Final[int] = 0  # Changed from float to int
    AWAY_FROM_FOOD: Final[int] = 0  # Changed from float to int


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
class AgentInfo:
    name: str
    requires_training: bool


@dataclass(frozen=True)
class ModelType:
    QLEARNING: Final[str] = "qlearning"
    ZIGZAG: Final[str] = "zigzag"
    BFS: Final[str] = "bfs"
    BFS_ZIGZAG: Final[str] = "bfs_zigzag"

    @classmethod
    def get_agent_types(cls) -> Dict[str, AgentInfo]:
        """Return a dictionary mapping agent type identifiers to their metadata"""
        return {
            cls.QLEARNING: AgentInfo(name="Q-Learning", requires_training=True),
            cls.ZIGZAG: AgentInfo(name="Zigzag Cycle", requires_training=False),
            cls.BFS: AgentInfo(name="Breadth-First Search", requires_training=False),
            cls.BFS_ZIGZAG: AgentInfo(
                name="BFS-Zigzag Hybrid", requires_training=False
            ),
        }

    @staticmethod
    def from_string(model_type: str) -> str:
        model_type = model_type.lower()
        if model_type in ModelType.get_agent_types():
            return model_type
        raise ValueError(f"Unknown agent type: {model_type}")

    @staticmethod
    def requires_training(agent_type: str) -> bool:
        """Check if the agent type requires training"""
        agent_type = ModelType.from_string(agent_type)
        return ModelType.get_agent_types()[agent_type].requires_training

    @staticmethod
    def get_display_name(agent_type: str) -> str:
        """Get the display name for an agent type"""
        agent_type = ModelType.from_string(agent_type)
        return ModelType.get_agent_types()[agent_type].name

    @staticmethod
    def _create_environment(render_mode: str, debug_mode: bool = False) -> Any:
        """Create a snake environment"""
        from snake_env import ImprovedSnakeEnv

        return ImprovedSnakeEnv(
            grid_size=SnakeConfig.DEFAULT_GRID_SIZE,
            block_size=SnakeConfig.DEFAULT_BLOCK_SIZE,
            render_mode=render_mode,
            debug_mode=debug_mode,
        )

    @staticmethod
    def create_agent(agent_type: str, debug_mode: bool = False) -> Any:
        """Create an agent instance of the specified type with debug mode setting"""
        agent_type = ModelType.from_string(agent_type)
        if agent_type == ModelType.QLEARNING:
            from agents.qlearning_agent import QLearningAgent

            return QLearningAgent.create(debug_mode=debug_mode)
        elif agent_type == ModelType.ZIGZAG:
            from agents.zigzag_agent import ZigzagAgent

            return ZigzagAgent.create(debug_mode=debug_mode)
        elif agent_type == ModelType.BFS:
            from agents.bfs_agent import BFSAgent

            return BFSAgent.create(debug_mode=debug_mode)
        elif agent_type == ModelType.BFS_ZIGZAG:
            from agents.bfs_zigzag_agent import BFSZigzagAgent

            return BFSZigzagAgent.create(debug_mode=debug_mode)
        raise ValueError(f"No agent implementation for agent type: {agent_type}")

    @staticmethod
    def extract_from_filename(filename: str) -> str:
        # First check if the filename is just the agent type itself
        if filename.lower() in ModelType.get_agent_types():
            return filename.lower()

        # Extract agent type from filename like "20250224230142_qlearning.pkl"
        pattern = r"_(\w+)(?:_|\.)"
        match = re.search(pattern, filename)
        if match:
            return match.group(1)

        raise ValueError(f"Could not extract agent type from filename: {filename}")


def create_default_environment(render_mode: str, debug_mode: bool = False) -> Any:
    """Create a default snake environment with standard settings"""
    # Import here to avoid circular import
    from snake_env import ImprovedSnakeEnv

    return ImprovedSnakeEnv(
        grid_size=SnakeConfig.DEFAULT_GRID_SIZE,
        block_size=SnakeConfig.DEFAULT_BLOCK_SIZE,
        render_mode=render_mode,
        debug_mode=debug_mode,
    )
