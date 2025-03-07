import os
import pickle
from collections import deque
from datetime import datetime
from typing import Any, List, Optional

from agents.base_agent import BaseAgent
from constants import Action, Point, SnakeActions, State
from utils.logging_utils import setup_logger


class BFSAgent(BaseAgent):
    def __init__(self, debug_mode: bool = False) -> None:
        self.grid_size: int = 0
        self.debug_enabled: bool = debug_mode
        self.first_action: bool = True
        self.last_action: Optional[Action] = None
        self.actions_taken: int = 0
        self.last_position: Optional[Point] = None
        self.logger, _ = setup_logger("BFSAgent", debug_mode)

    @property
    def requires_training(self) -> bool:
        return False

    def choose_action(self, state: State) -> Action:
        pass

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
            self._log_surroundings(snake_head, env.snake)

        if self.grid_size != env.grid_size:
            self.grid_size = env.grid_size
            self.logger.info(f"Setting grid size to {self.grid_size}")

        # Find path to food
        path_to_food = self._find_path_to_food(snake_head, food, snake_body)

        if not path_to_food:
            self.logger.warning("No direct path to food found")
            # Fall back to survival mode - find the path to the area with the most space
            action = self._find_survival_path(snake_head, snake_body)
            self.logger.info(f"Taking survival action: {action}")
        else:
            # Check if taking the next step in the path is safe
            next_pos = path_to_food[1]  # 0 is current position, 1 is next position

            # Simulate taking the step
            virtual_snake = [next_pos] + env.snake[:-1]

            # Check if we can still reach our tail after taking this step
            if self._is_safe_move(next_pos, virtual_snake):
                # Get action to move to the next position in the path
                action = self._get_action_for_move(snake_head, next_pos)
                self.logger.debug(f"Taking path action to food: {action}")
            else:
                self.logger.warning(f"Path to food exists but is unsafe after eating")
                # Fall back to survival mode
                action = self._find_survival_path(snake_head, snake_body)
                self.logger.debug(f"Taking survival action instead: {action}")

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

    def _find_path_to_food(
        self, start: Point, target: Point, snake_body: List[Point]
    ) -> List[Point]:
        """BFS to find shortest path to food"""
        queue = deque([(start, [start])])
        visited = set([start])

        while queue:
            (x, y), path = queue.popleft()

            if (x, y) == target:
                self.logger.debug(f"Found path to food with length {len(path)}")
                return path

            # Check all four adjacent cells
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)

                # Skip if out of bounds or part of snake body or already visited
                if (
                    nx < 0
                    or nx >= self.grid_size
                    or ny < 0
                    or ny >= self.grid_size
                    or next_pos in snake_body
                    or next_pos in visited
                ):
                    continue

                visited.add(next_pos)
                new_path = path + [next_pos]
                queue.append((next_pos, new_path))

        self.logger.warning(f"No path found from {start} to {target}")
        return []

    def _is_safe_move(self, next_pos: Point, snake: List[Point]) -> bool:
        """Check if move is safe using flood fill to ensure snake won't trap itself"""
        # Create a virtual snake after the move (head at next_pos)
        virtual_head = next_pos
        virtual_body = snake[1:]  # Exclude head

        # If snake would eat food, it doesn't shrink
        # So we need to make sure there's enough space for a longer snake
        target_space = len(snake) + 1

        # Count accessible cells using flood fill
        accessible_cells = self._flood_fill(virtual_head, virtual_body)

        if self.debug_enabled:
            self.logger.info(
                f"Accessible cells after move: {accessible_cells}, need {target_space}"
            )

        # Move is safe if we can access enough cells
        return accessible_cells >= target_space

    def _flood_fill(self, start: Point, obstacles: List[Point]) -> int:
        """Count accessible cells from start position using flood fill"""
        queue = deque([start])
        visited = set([start])
        count = 0

        while queue:
            x, y = queue.popleft()
            count += 1

            # Check all four adjacent cells
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)

                # Skip if out of bounds or obstacle or already visited
                if (
                    nx < 0
                    or nx >= self.grid_size
                    or ny < 0
                    or ny >= self.grid_size
                    or next_pos in obstacles
                    or next_pos in visited
                ):
                    continue

                visited.add(next_pos)
                queue.append(next_pos)

        return count

    def _find_survival_path(self, head: Point, snake_body: List[Point]) -> Action:
        """Find the direction with the most open space"""
        best_action = None
        max_space = -1

        # Try all four directions
        for action, (dx, dy) in {
            SnakeActions.UP: (0, -1),
            SnakeActions.RIGHT: (1, 0),
            SnakeActions.DOWN: (0, 1),
            SnakeActions.LEFT: (-1, 0),
        }.items():
            nx, ny = head[0] + dx, head[1] + dy
            next_pos = (nx, ny)

            # Skip invalid moves
            if (
                nx < 0
                or nx >= self.grid_size
                or ny < 0
                or ny >= self.grid_size
                or next_pos in snake_body
            ):
                continue

            # Count accessible space
            virtual_snake = [next_pos] + snake_body[:-1]  # Virtual snake after move
            space = self._flood_fill(next_pos, virtual_snake)

            self.logger.info(f"Direction {action}: {space} accessible cells")

            if space > max_space:
                max_space = space
                best_action = action

        # If no valid moves, just go up (will crash, but it's the best we can do)
        return best_action if best_action else SnakeActions.UP

    def _get_action_for_move(self, current: Point, next_pos: Point) -> Action:
        """Get the action needed to move from current to next position"""
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]

        if dx == 1:
            return SnakeActions.RIGHT
        elif dx == -1:
            return SnakeActions.LEFT
        elif dy == 1:
            return SnakeActions.DOWN
        else:
            return SnakeActions.UP

    def _log_surroundings(self, position: Point, snake: List[Point]) -> None:
        """Log the surroundings of the snake head for debugging"""
        x, y = position
        surrounding = [
            ((x, y - 1), "UP"),
            ((x + 1, y), "RIGHT"),
            ((x, y + 1), "DOWN"),
            ((x - 1, y), "LEFT"),
        ]

        surroundings_info = []
        for pos, direction in surrounding:
            x_pos, y_pos = pos
            if (
                x_pos < 0
                or x_pos >= self.grid_size
                or y_pos < 0
                or y_pos >= self.grid_size
            ):
                status = "WALL"
            elif pos in snake:
                status = "SNAKE"
            else:
                status = "EMPTY"

            surroundings_info.append(f"{direction}: {pos} - {status}")

        self.logger.info("Surroundings:")
        for info in surroundings_info:
            self.logger.info(f"  {info}")

    def train(self, env: Any, num_episodes: int, suffix: Optional[str]) -> str:
        """
        BFS agent doesn't require training as it's a deterministic algorithm.
        This method is implemented to satisfy the BaseAgent interface but it only initializes the grid size.
        """
        self.grid_size = env.grid_size
        self.logger.info(
            f"BFS agent doesn't require training (grid size: {self.grid_size})"
        )

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{timestamp}_bfs{f'_{suffix}' if suffix else ''}"

    def save(self, filename: str) -> None:
        """
        BFS agent doesn't require saving a model as it's a deterministic algorithm.
        This method is implemented to satisfy the BaseAgent interface.
        """
        self.logger.info("BFS agent doesn't require saving a model (no-op)")
        # Still create the file to avoid errors, but it contains minimal data
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump({"grid_size": self.grid_size, "algorithm": "bfs"}, f)

    def load(self, filename: str) -> None:
        """
        BFS agent doesn't require loading a model as it's a deterministic algorithm.
        This method is implemented to satisfy the BaseAgent interface.
        """
        self.logger.info("BFS agent doesn't require loading a model (no-op)")
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.grid_size = data.get("grid_size", 0)
                self.logger.info(f"Retrieved grid size: {self.grid_size}")
        except Exception as e:
            self.logger.warning(
                f"Couldn't load file, but this is non-critical: {str(e)}"
            )

    @classmethod
    def create(cls, debug_mode: bool) -> "BFSAgent":
        return cls(debug_mode=debug_mode)
