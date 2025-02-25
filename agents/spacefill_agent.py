import datetime
import sys
from typing import List, Tuple

from agents.base_agent import BaseAgent
from constants import Action, Point, SnakeActions, State


class SpaceFillingAgent(BaseAgent):
    """
    A baseline agent that follows a predetermined zigzag path through the grid.
    The agent moves left across the top row, down one cell, right across the second row,
    down one cell, left across the third row, etc. This creates a hamiltonian path that
    visits every cell exactly once before repeating, which guarantees the snake will never
    collide with itself if it follows the pattern properly.
    """

    def __init__(self) -> None:
        self.grid_size = 0
        self.path: List[Point] = []
        self.current_pos_index = 0
        self.initialized = False

    def _generate_path(self, grid_size: int) -> List[Point]:
        """Generate a zigzag path that visits every cell in the grid."""
        path = []
        for y in range(grid_size):
            # If we're on an even row, go left to right
            if y % 2 == 0:
                for x in range(grid_size):
                    path.append((x, y))
            # If we're on an odd row, go right to left
            else:
                for x in range(grid_size - 1, -1, -1):
                    path.append((x, y))
        return path

    def _find_closest_path_position(self, snake_head: Point, path: List[Point]) -> int:
        """Find the index of the path position closest to the snake's head."""
        if not path:
            return 0

        # First check if the head is exactly on a path position
        try:
            return path.index(snake_head)
        except ValueError:
            pass

        # Otherwise find the closest point on the path
        closest_idx = 0
        min_dist = float("inf")

        for i, pos in enumerate(path):
            dist = abs(snake_head[0] - pos[0]) + abs(snake_head[1] - pos[1])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx

    def _get_next_safe_action(self, env, head: Point, target: Point) -> Action:
        """
        Determine the best action to move from head to target position
        that avoids immediate collisions.
        """
        # Calculate the direction we want to move
        if target[0] > head[0]:
            desired_action = SnakeActions.RIGHT
        elif target[0] < head[0]:
            desired_action = SnakeActions.LEFT
        elif target[1] > head[1]:
            desired_action = SnakeActions.DOWN
        else:
            desired_action = SnakeActions.UP

        # Check if the desired action is safe
        if self._is_safe_action(env, desired_action):
            return desired_action

        # If not safe, try other directions in order of preference
        # (based on getting closer to the target)
        dx = target[0] - head[0]
        dy = target[1] - head[1]

        actions_to_try = []

        # Prioritize horizontal or vertical movement based on which axis
        # has the greater distance to the target
        if abs(dx) > abs(dy):
            # Prioritize horizontal movement
            if dx > 0:
                actions_to_try = [
                    SnakeActions.RIGHT,
                    SnakeActions.DOWN,
                    SnakeActions.UP,
                    SnakeActions.LEFT,
                ]
            else:
                actions_to_try = [
                    SnakeActions.LEFT,
                    SnakeActions.DOWN,
                    SnakeActions.UP,
                    SnakeActions.RIGHT,
                ]
        else:
            # Prioritize vertical movement
            if dy > 0:
                actions_to_try = [
                    SnakeActions.DOWN,
                    SnakeActions.RIGHT,
                    SnakeActions.LEFT,
                    SnakeActions.UP,
                ]
            else:
                actions_to_try = [
                    SnakeActions.UP,
                    SnakeActions.RIGHT,
                    SnakeActions.LEFT,
                    SnakeActions.DOWN,
                ]

        # Try each action in order of preference
        for action in actions_to_try:
            if self._is_safe_action(env, action):
                return action

        # If no safe action is found, return the original desired action as a last resort
        return desired_action

    def _is_safe_action(self, env, action: Action) -> bool:
        """Check if an action is safe (won't result in immediate collision)."""
        head_x, head_y = env.snake[0]

        # Determine new head position after the action
        if action == SnakeActions.UP:
            new_head = (head_x, head_y - 1)
        elif action == SnakeActions.DOWN:
            new_head = (head_x, head_y + 1)
        elif action == SnakeActions.LEFT:
            new_head = (head_x - 1, head_y)
        elif action == SnakeActions.RIGHT:
            new_head = (head_x + 1, head_y)
        else:
            return False

        # Check if the new position is outside the grid
        if (
            new_head[0] < 0
            or new_head[0] >= env.grid_size
            or new_head[1] < 0
            or new_head[1] >= env.grid_size
        ):
            return False

        # Check if the new position collides with the snake's body
        if new_head in env.snake[:-1]:  # Exclude the tail
            return False

        # Check if the action is a 180-degree turn (not allowed)
        if (
            (env.direction == SnakeActions.UP and action == SnakeActions.DOWN)
            or (env.direction == SnakeActions.DOWN and action == SnakeActions.UP)
            or (env.direction == SnakeActions.LEFT and action == SnakeActions.RIGHT)
            or (env.direction == SnakeActions.RIGHT and action == SnakeActions.LEFT)
        ):
            return False

        return True

    def choose_action(self, state: State) -> Action:
        from snake_env import ImprovedSnakeEnv

        # Get the environment to access snake and grid information
        env = self._get_environment_from_state()

        if not env:
            # Fallback action if we can't get the environment
            return SnakeActions.RIGHT

        # Initialize path if not already done
        if not self.initialized or self.grid_size != env.grid_size:
            self.grid_size = env.grid_size
            self.path = self._generate_path(self.grid_size)
            self.current_pos_index = self._find_closest_path_position(
                env.snake[0], self.path
            )
            self.initialized = True

        # Get current snake head position
        head = env.snake[0]

        # Find the next position in the path
        next_index = (self.current_pos_index + 1) % len(self.path)
        next_position = self.path[next_index]

        # If we're not exactly on the current path position, try to get back on track
        if head != self.path[self.current_pos_index]:
            current_on_path_position = self.path[self.current_pos_index]
            return self._get_next_safe_action(env, head, current_on_path_position)

        # Update current position index since we're on the correct path
        self.current_pos_index = next_index

        # Get a safe action to move to the next position
        return self._get_next_safe_action(env, head, next_position)

    def _get_environment_from_state(self):
        """Get the environment instance from the current context."""
        import inspect

        frame = inspect.currentframe()
        while frame:
            if "env" in frame.f_locals and hasattr(frame.f_locals["env"], "snake"):
                return frame.f_locals["env"]
            if "self" in frame.f_locals and hasattr(frame.f_locals["self"], "snake"):
                return frame.f_locals["self"]
            frame = frame.f_back
        return None

    def train(self, env, num_episodes: int, suffix=None) -> str:
        """Space filling agent doesn't need training."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_name = f"{timestamp}_spacefill"
        if suffix:
            model_name = f"{model_name}_{suffix}"
        return model_name

    def save(self, filename: str) -> None:
        """Space filling agent has no state to save."""
        pass

    def load(self, filename: str) -> None:
        """Space filling agent has no state to load."""
        pass
