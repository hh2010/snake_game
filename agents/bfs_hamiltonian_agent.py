import logging
import os
import pickle
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from agents.base_agent import BaseAgent
from agents.bfs_agent import BFSAgent
from agents.hamiltonian_agent import HamiltonianAgent
from constants import Action, Point, SnakeActions, State


class BFSHamiltonianAgent(BaseAgent):
    def __init__(self, enable_logging: bool) -> None:
        self.grid_size: int = 0
        self.debug_enabled: bool = enable_logging
        self.actions_taken: int = 0
        self.last_position: Optional[Point] = None
        self.bfs_mode: bool = True  # Track which mode we're in - BFS or Hamiltonian

        # Initialize component agents
        self.bfs_agent = BFSAgent(enable_logging=enable_logging)
        self.hamiltonian_agent = HamiltonianAgent(enable_logging=enable_logging)

        self._setup_logging(enable_logging)

    def _setup_logging(self, enable_logging: bool) -> None:
        if not enable_logging:
            self.logger = logging.getLogger("NullLogger")
            self.logger.addHandler(logging.NullHandler())
            return

        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/bfs_hamiltonian_agent_{timestamp}.log"

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
        )
        self.logger = logging.getLogger("BFSHamiltonianAgent")
        self.logger.info(
            f"Initializing BFS-Hamiltonian Hybrid Agent - Log file: {log_filename}"
        )

    def choose_action(self, state: State) -> Action:
        from snake_env import ImprovedSnakeEnv

        env = self._get_env_from_state()

        if not env:
            self.logger.error("Could not access environment from state")
            return SnakeActions.UP

        snake_head = env.snake[0]
        snake_body = env.snake[1:]
        food = env.food

        if self.debug_enabled and self.last_position != snake_head:
            self.logger.info(f"Snake head at {snake_head}, Food at {food}")
            self.last_position = snake_head

        if self.grid_size != env.grid_size:
            self.grid_size = env.grid_size
            self.logger.info(f"Setting grid size to {self.grid_size}")

            # Make sure both component agents have the grid size set
            self.bfs_agent.grid_size = self.grid_size
            self.hamiltonian_agent.grid_size = self.grid_size

            # Generate Hamiltonian cycle when grid size changes
            self.hamiltonian_agent._generate_hamiltonian_cycle()

        # Try BFS path to food first
        path_to_food = self.bfs_agent._find_path_to_food(snake_head, food, snake_body)

        if not path_to_food:
            # No direct path to food - switch to Hamiltonian mode
            if self.bfs_mode:
                self.logger.info("No path to food found. Switching to Hamiltonian mode")
                self.bfs_mode = False

            action = self.hamiltonian_agent._get_next_action_in_cycle(
                snake_head, snake_body
            )
            self.logger.info(f"Using Hamiltonian action: {action}")

        else:
            # Path to food exists, check if it's safe
            next_pos = path_to_food[1]  # 0 is current position, 1 is next position
            virtual_snake = [next_pos] + env.snake[:-1]

            # Check safety with flood fill
            if self.bfs_agent._is_safe_move(next_pos, virtual_snake):
                # Safe to use BFS path
                if not self.bfs_mode:
                    self.logger.info("Safe path found. Switching to BFS mode")
                    self.bfs_mode = True

                action = self.bfs_agent._get_action_for_move(snake_head, next_pos)
                self.logger.info(f"Using BFS action: {action}")

            else:
                # Not safe - switch to Hamiltonian mode
                if self.bfs_mode:
                    self.logger.info(
                        "Path to food exists but is unsafe. Switching to Hamiltonian mode"
                    )
                    self.bfs_mode = False

                action = self.hamiltonian_agent._get_next_action_in_cycle(
                    snake_head, snake_body
                )
                self.logger.info(f"Using Hamiltonian action: {action}")

        self.actions_taken += 1
        if self.actions_taken % 100 == 0:
            self.logger.info(
                f"Action #{self.actions_taken}: {action} from {snake_head}"
            )

        return action

    def _get_env_from_state(self) -> Any:
        import inspect

        frame = inspect.currentframe()
        while frame:
            if "env" in frame.f_locals:
                return frame.f_locals["env"]
            frame = frame.f_back
        return None

    def train(self, env: Any, num_episodes: int, suffix: Optional[str]) -> str:
        self.grid_size = env.grid_size
        self.logger.info(
            f"Training BFS-Hamiltonian agent with grid size {self.grid_size}"
        )

        # Set up both component agents
        self.bfs_agent.grid_size = self.grid_size
        self.hamiltonian_agent.grid_size = self.grid_size
        self.hamiltonian_agent._generate_hamiltonian_cycle()

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{timestamp}_bfs_hamiltonian{f'_{suffix}' if suffix else ''}"

    def save(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "grid_size": self.grid_size,
                    "hamiltonian_cycle": self.hamiltonian_agent.hamiltonian_cycle,
                    "hamiltonian_map": self.hamiltonian_agent.hamiltonian_map,
                },
                f,
            )
        self.logger.info(f"Model saved to {filename}")

    def load(self, filename: str) -> None:
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.grid_size = data["grid_size"]
                self.hamiltonian_agent.grid_size = self.grid_size
                self.hamiltonian_agent.hamiltonian_cycle = data["hamiltonian_cycle"]
                self.hamiltonian_agent.hamiltonian_map = data["hamiltonian_map"]
                self.bfs_agent.grid_size = self.grid_size

            self.logger.info(f"Model loaded from {filename}")
            self.logger.info(f"Grid size: {self.grid_size}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    @classmethod
    def create(cls, enable_logging: bool) -> "BFSHamiltonianAgent":
        return cls(enable_logging=enable_logging)
