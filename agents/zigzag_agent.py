import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast

from agents.base_agent import BaseAgent
from constants import Action, Point, SnakeActions, State
from utils.logging_utils import setup_logger


class ZigzagAgent(BaseAgent):
    def __init__(self, debug_mode: bool = False) -> None:
        self.zigzag_map: Dict[Point, int] = {}
        self.zigzag_cycle: List[Point] = []
        self.grid_size: int = 0
        self.debug_enabled: bool = debug_mode
        self.first_action: bool = True
        self.last_action: Optional[Action] = None
        self.cycles_completed: int = 0
        self.actions_taken: int = 0
        self.last_position: Optional[Point] = None
        self.debug_cycle_graphic: bool = debug_mode
        self.logger, _ = setup_logger("ZigzagAgent", debug_mode)

    @property
    def requires_training(self) -> bool:
        return False

    def choose_action(self, state: State) -> Action:
        from snake_env import ImprovedSnakeEnv

        env = self._get_env_from_state()

        if not env:
            self.logger.error("Could not access environment from state")
            return env.direction if env else SnakeActions.UP

        snake_head = env.snake[0]

        if self.debug_enabled and self.last_position != snake_head:
            self.logger.debug(
                f"Snake head at {snake_head}, Current direction: {env.direction}"
            )
            self.last_position = snake_head
            self._log_surroundings(snake_head, env.snake)

        if self.grid_size != env.grid_size or not self.zigzag_cycle:
            self.grid_size = env.grid_size
            self.logger.info(
                f"Generating new Zigzag cycle for grid size {self.grid_size}"
            )
            self._generate_zigzag_cycle()

            if self.debug_enabled:
                self.logger.debug(
                    f"Generated Zigzag cycle for grid size {self.grid_size}"
                )
                self.logger.debug(f"Cycle length: {len(self.zigzag_cycle)}")
                self.logger.debug(
                    f"Cycle starts at {self.zigzag_cycle[0]} and ends at {self.zigzag_cycle[-1]}"
                )

                if self.debug_cycle_graphic:
                    self._log_cycle_graphic()

                self._log_next_few_steps(snake_head)
                self.first_action = True

        if snake_head not in self.zigzag_map:
            self.logger.warning(f"Snake head {snake_head} not in Zigzag cycle!")
            self.logger.warning(
                f"Valid cycle points: {len(self.zigzag_map)} of {self.grid_size * self.grid_size}"
            )

            sample_points = list(self.zigzag_map.keys())[:5]
            self.logger.warning(f"Sample cycle points: {sample_points}")

            action = self._get_closest_cycle_action(snake_head, env.snake[1:])
            self.last_action = action
            self.actions_taken += 1
            self.logger.info(f"Taking recovery action: {action}")
            return action

        action = self._get_next_action_in_cycle(snake_head, env.snake[1:])

        if self.debug_enabled and self.first_action:
            self.logger.info(f"First action: {action}")
            self.first_action = False

        self.last_action = action
        self.actions_taken += 1

        if self.actions_taken % 100 == 0:
            self.logger.debug(
                f"Action #{self.actions_taken}: {action} from {snake_head}"
            )

        return action

    def _log_cycle_graphic(self) -> None:
        if not self.zigzag_cycle:
            return

        self.logger.debug("Zigzag Cycle Visualization:")
        self.logger.debug("--------------------------------")

        grid: List[List[Optional[int]]] = []
        for _ in range(self.grid_size):
            grid.append([None] * self.grid_size)

        for i, (x, y) in enumerate(self.zigzag_cycle):
            grid[y][x] = i

        for y, row in enumerate(grid):
            line = f"Row {y}: "
            for cell in row:
                if cell is None:
                    line += "XXX "
                else:
                    line += f"{cell:03} "
            self.logger.debug(line)

        first = self.zigzag_cycle[0]
        last = self.zigzag_cycle[-1]
        self.logger.debug(f"Connection: {last} -> {first}")

        if abs(last[0] - first[0]) + abs(last[1] - first[1]) == 1:
            self.logger.debug("Cycle is valid: last point connects to first point")
        else:
            self.logger.error(
                f"INVALID CYCLE: last point {last} and first point {first} aren't connected!"
            )

    def _log_surroundings(self, position: Point, snake: List[Point]) -> None:
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
                if pos in self.zigzag_map:
                    cycle_idx = self.zigzag_map[pos]
                    head_idx = self.zigzag_map.get(position, -1)
                    status = f"CYCLE({cycle_idx})"
                    if head_idx != -1:
                        next_idx = (head_idx + 1) % len(self.zigzag_cycle)
                        if cycle_idx == next_idx:
                            status += " (NEXT)"
                else:
                    status = "EMPTY"

            surroundings_info.append(f"{direction}: {pos} - {status}")

        self.logger.debug("Surroundings:")
        for info in surroundings_info:
            self.logger.debug(f"  {info}")

    def _log_next_few_steps(self, current_pos: Point) -> None:
        if current_pos not in self.zigzag_map:
            self.logger.warning(f"Cannot show next steps - {current_pos} not in cycle")
            return

        curr_idx = self.zigzag_map[current_pos]
        self.logger.debug(
            f"Current position {current_pos} is at index {curr_idx} in the cycle"
        )

        self.logger.debug("Next 10 steps in the cycle:")
        for i in range(1, 11):
            next_idx = (curr_idx + i) % len(self.zigzag_cycle)
            next_pos = self.zigzag_cycle[next_idx]
            self.logger.debug(f"  Step {i}: {next_pos} (idx={next_idx})")

    def _get_env_from_state(self) -> Any:
        import inspect

        frame = inspect.currentframe()
        while frame:
            if "env" in frame.f_locals:
                return frame.f_locals["env"]
            frame = frame.f_back
        return None

    def _get_closest_cycle_action(self, pos: Point, snake_body: List[Point]) -> Action:
        best_dist = float("inf")
        closest_point = None

        for point in self.zigzag_cycle:
            if point in snake_body:
                continue

            dist = abs(pos[0] - point[0]) + abs(pos[1] - point[1])
            if dist < best_dist:
                best_dist = dist
                closest_point = point

        self.logger.info(
            f"Closest cycle point to {pos} is {closest_point} (dist={best_dist})"
        )

        if closest_point:
            possible_moves = {
                SnakeActions.UP: (pos[0], pos[1] - 1),
                SnakeActions.DOWN: (pos[0], pos[1] + 1),
                SnakeActions.LEFT: (pos[0] - 1, pos[1]),
                SnakeActions.RIGHT: (pos[0] + 1, pos[1]),
            }

            valid_moves = {}
            for action, new_pos in possible_moves.items():
                x, y = new_pos
                if (
                    0 <= x < self.grid_size
                    and 0 <= y < self.grid_size
                    and new_pos not in snake_body
                ):
                    valid_moves[action] = new_pos

            self.logger.info(f"Valid moves from {pos}: {list(valid_moves.keys())}")

            if not valid_moves:
                self.logger.warning("No valid moves available!")
                return SnakeActions.UP

            best_action: Optional[str] = None
            best_move_dist = float("inf")

            for action, new_pos in valid_moves.items():
                move_dist = abs(new_pos[0] - closest_point[0]) + abs(
                    new_pos[1] - closest_point[1]
                )
                if move_dist < best_move_dist:
                    best_move_dist = move_dist
                    best_action = action

            self.logger.info(f"Best action to get to {closest_point}: {best_action}")
            return best_action if best_action is not None else SnakeActions.UP

        self.logger.error("No closest point found - defaulting to UP")
        return SnakeActions.UP

    def _generate_zigzag_cycle(self) -> None:
        self.zigzag_map = {}
        self.zigzag_cycle = []

        if self.grid_size % 2 != 0:
            self.logger.info(
                f"Using odd-sized grid algorithm for size {self.grid_size}"
            )
            self._generate_odd_sized_zigzag_cycle()
            return

        self.logger.info(
            f"Using even-sized grid algorithm with modified pattern for size {self.grid_size}"
        )

        cycle = []

        start_point = (1, 0)
        cycle.append(start_point)

        # Step 1: Go right from (1, 0) to (grid_size-1, 0) for the first row
        for x in range(2, self.grid_size):
            cycle.append((x, 0))

        # Step 2: Zigzag through all rows from row 1 to grid_size-1,
        # but only using columns 1 to grid_size-1
        for y in range(1, self.grid_size):
            row = []
            for x in range(1, self.grid_size):
                row.append((x, y))

            if y % 2 != 0:  # Odd rows go right to left
                row.reverse()

            cycle.extend(row)

        # Step 3: Now come up through the leftmost column (from bottom to top)
        for y in range(self.grid_size - 1, -1, -1):
            cycle.append((0, y))

        self.zigzag_cycle = cycle

        for i, point in enumerate(cycle):
            self.zigzag_map[point] = i

        self.logger.info(f"Generated even-sized Zigzag cycle with {len(cycle)} points")
        self.logger.info(f"Starting point: {self.zigzag_cycle[0]}")
        self.logger.info(f"Ending point: {self.zigzag_cycle[-1]}")

        self._validate_cycle()

    def _generate_odd_sized_zigzag_cycle(self) -> None:
        if self.grid_size == 1:
            self.zigzag_cycle = [(0, 0)]
            self.zigzag_map = {(0, 0): 0}
            self.logger.info("Special case: grid size 1")
            return

        cycle = []

        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                row.append((x, y))

            if y % 2 == 1:
                row.reverse()

            cycle.extend(row)

        last_point = cycle[-1]
        first_point = cycle[0]

        if (
            abs(last_point[0] - first_point[0]) + abs(last_point[1] - first_point[1])
            != 1
        ):
            self.logger.info(
                f"Fixing odd-sized grid cycle: last={last_point}, first={first_point}"
            )

            even_size = self.grid_size - 1
            cycle = []

            # Handle the main even-sized sub-grid (top-left portion)
            for y in range(even_size):
                row = []
                for x in range(even_size):
                    row.append((x, y))

                if y % 2 == 1:
                    row.reverse()

                cycle.extend(row)

            # Add the rightmost column from top to bottom
            for y in range(even_size):
                cycle.append((even_size, y))

            # Add the bottom row from right to left
            for x in range(even_size, -1, -1):
                cycle.append((x, even_size))

            # Add any remaining points if needed
            all_points = set(cycle)
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if (x, y) not in all_points:
                        inserted = False
                        for i in range(len(cycle)):
                            p1 = cycle[i]
                            p2 = cycle[(i + 1) % len(cycle)]
                            if (
                                abs(p1[0] - x) + abs(p1[1] - y) == 1
                                and abs(p2[0] - x) + abs(p2[1] - y) == 1
                            ):
                                cycle.insert(i + 1, (x, y))
                                inserted = True
                                self.logger.info(
                                    f"Inserted point ({x}, {y}) into cycle at position {i+1}"
                                )
                                break
                        if not inserted:
                            self.logger.warning(
                                f"Could not insert point ({x}, {y}) into cycle"
                            )

        self.zigzag_cycle = cycle

        for i, point in enumerate(cycle):
            self.zigzag_map[point] = i

        self.logger.info(f"Generated odd-sized Zigzag cycle with {len(cycle)} points")

        self._validate_cycle()

    def _validate_cycle(self) -> None:
        # Check that all grid positions are included
        points_in_cycle = set(self.zigzag_cycle)
        missing_points = []

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (x, y) not in points_in_cycle:
                    missing_points.append((x, y))

        if missing_points:
            self.logger.error(
                f"ERROR: {len(missing_points)} positions missing from Zigzag cycle!"
            )
            for p in missing_points[:5]:
                self.logger.error(f"  Missing: {p}")

        # Check that each step in the cycle is valid (adjacent points)
        invalid_steps = []
        for i in range(len(self.zigzag_cycle) - 1):
            p1 = self.zigzag_cycle[i]
            p2 = self.zigzag_cycle[i + 1]
            if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) != 1:
                invalid_steps.append((i, p1, p2))

        if invalid_steps:
            self.logger.error(
                f"ERROR: {len(invalid_steps)} invalid cycle steps detected!"
            )
            for idx, p1, p2 in invalid_steps[:5]:
                self.logger.error(f"  Invalid step at index {idx}: {p1} -> {p2}")

        # Check that the last point connects back to the first
        p1 = self.zigzag_cycle[-1]
        p2 = self.zigzag_cycle[0]
        if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) != 1:
            self.logger.error(
                f"ERROR: Last point {p1} and first point {p2} are not connected! Distance: {abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])}"
            )
            self._try_fix_cycle()
        else:
            self.logger.info(f"Zigzag cycle validation complete - cycle is valid")
            self.logger.info(f"Starting point: {self.zigzag_cycle[0]}")
            self.logger.info(f"Ending point: {self.zigzag_cycle[-1]}")

    def _try_fix_cycle(self) -> None:
        if not self.zigzag_cycle:
            return

        first = self.zigzag_cycle[0]
        last = self.zigzag_cycle[-1]

        self.logger.info(f"Attempting to fix cycle: connect {last} to {first}")

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (x, y) in self.zigzag_cycle:
                    continue

                if (
                    abs(x - first[0]) + abs(y - first[1]) == 1
                    and abs(x - last[0]) + abs(y - last[1]) == 1
                ):
                    self.logger.info(f"Found linking point: ({x}, {y})")

                    self.zigzag_cycle.append((x, y))

                    for i, point in enumerate(self.zigzag_cycle):
                        self.zigzag_map[point] = i

                    self.logger.info("Cycle fixed successfully!")
                    return

        self.logger.error("Could not find a way to fix the cycle")

    def _get_next_action_in_cycle(
        self, current_pos: Point, snake_body: List[Point]
    ) -> Action:
        if not self.zigzag_cycle or current_pos not in self.zigzag_map:
            self.logger.error(
                f"Position {current_pos} not in Zigzag cycle! Defaulting to UP."
            )
            return SnakeActions.UP

        curr_idx = self.zigzag_map[current_pos]

        next_idx = (curr_idx + 1) % len(self.zigzag_cycle)
        next_pos = self.zigzag_cycle[next_idx]

        if self.debug_enabled and self.first_action:
            self.logger.info(
                f"Starting position in cycle: {curr_idx}/{len(self.zigzag_cycle)}"
            )
            self.logger.info(f"Next position will be: {next_pos} (index {next_idx})")
            self.first_action = False

        if next_idx == 0 and curr_idx == len(self.zigzag_cycle) - 1:
            self.cycles_completed += 1
            self.logger.info(f"Completed cycle #{self.cycles_completed}")

        if (
            next_pos[0] < 0
            or next_pos[0] >= self.grid_size
            or next_pos[1] < 0
            or next_pos[1] >= self.grid_size
            or next_pos in snake_body
        ):
            self.logger.error(
                f"Next position {next_pos} in cycle is not safe! This should never happen."
            )

        x1, y1 = current_pos
        x2, y2 = next_pos

        if x2 < x1:
            return SnakeActions.LEFT
        elif x2 > x1:
            return SnakeActions.RIGHT
        elif y2 < y1:
            return SnakeActions.UP
        else:
            return SnakeActions.DOWN

    def train(self, env: Any, num_episodes: int, suffix: Optional[str]) -> str:
        """
        Zigzag agent doesn't require training as it's a deterministic algorithm.
        This method is implemented to satisfy the BaseAgent interface but just logs a message.
        """
        self.logger.info("Zigzag agent doesn't require training")
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{timestamp}_zigzag{f'_{suffix}' if suffix else ''}"

    def save(self, filename: str) -> None:
        """
        Zigzag agent doesn't need to save any data.
        This method is implemented to satisfy the BaseAgent interface.
        """
        self.logger.info("Zigzag agent doesn't need to save any data")

    def load(self, filename: str) -> None:
        """
        Zigzag agent doesn't need to load any data.
        This method is implemented to satisfy the BaseAgent interface.
        """
        self.logger.info("Zigzag agent doesn't need to load any data")

    @classmethod
    def create(cls, debug_mode: bool = False) -> "ZigzagAgent":
        return cls(debug_mode=debug_mode)
