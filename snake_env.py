import random
import time
from typing import List, Optional, Tuple, cast

# pylint: disable=no-member
import pygame

Point = Tuple[int, int]
State = Tuple[int, int, int, int, int, int, int, int]
Action = str


class ImprovedSnakeEnv:
    def __init__(
        self, grid_size: int = 15, block_size: int = 40, render_mode: str = "none"
    ) -> None:
        self.grid_size = grid_size
        self.block_size = block_size
        self.render_mode = render_mode
        self.score = 0
        self.start_time = time.time()
        self.snake: List[Point] = []
        self.food: Point = (0, 0)
        self.direction: Action = "UP"
        self.done = False
        self.window_size: int = 0
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self.reset()

        if self.render_mode == "human":
            pygame.init()
            self.window_size = self.grid_size * self.block_size
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

    def reset(self) -> State:
        self.snake = [
            (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
        ]
        self.food = (
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1),
        )
        self.direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        self.done = False
        self.score = 0
        self.start_time = time.time()
        return self.get_state()

    def get_state(self) -> State:
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        food_left = 1 if food_x < head_x else 0
        food_right = 1 if food_x > head_x else 0
        food_up = 1 if food_y < head_y else 0
        food_down = 1 if food_y > head_y else 0

        danger_left = 1 if (head_x - 1 < 0 or (head_x - 1, head_y) in self.snake) else 0
        danger_right = (
            1
            if (head_x + 1 >= self.grid_size or (head_x + 1, head_y) in self.snake)
            else 0
        )
        danger_up = 1 if (head_y - 1 < 0 or (head_x, head_y - 1) in self.snake) else 0
        danger_down = (
            1
            if (head_y + 1 >= self.grid_size or (head_x, head_y + 1) in self.snake)
            else 0
        )

        return (
            food_left,
            food_right,
            food_up,
            food_down,
            danger_left,
            danger_right,
            danger_up,
            danger_down,
        )

    def step(self, action: Action) -> Tuple[State, int, bool]:
        head_x, head_y = self.snake[0]
        new_head = None

        if action == "UP":
            new_head = (head_x, head_y - 1)
        elif action == "DOWN":
            new_head = (head_x, head_y + 1)
        elif action == "LEFT":
            new_head = (head_x - 1, head_y)
        elif action == "RIGHT":
            new_head = (head_x + 1, head_y)

        # Safety check in case of invalid action
        if new_head is None:
            self.done = True
            return self.get_state(), -10, self.done

        if (
            new_head[0] < 0
            or new_head[0] >= self.grid_size
            or new_head[1] < 0
            or new_head[1] >= self.grid_size
            or new_head in self.snake
        ):
            self.done = True
            return self.get_state(), -10, self.done

        self.snake.insert(0, new_head)

        if new_head == self.food:
            reward = 10
            self.score += 10
            self.food = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
        else:
            self.snake.pop()
            old_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = 1 if new_dist < old_dist else -1

        return self.get_state(), reward, self.done

    def render(self) -> None:
        if (
            self.render_mode == "none"
            or self.screen is None
            or self.font is None
            or self.clock is None
        ):
            return

        screen = cast(pygame.Surface, self.screen)
        font = cast(pygame.font.Font, self.font)
        clock = cast(pygame.time.Clock, self.clock)

        screen.fill((0, 0, 0))

        for segment in self.snake:
            pygame.draw.rect(
                screen,
                (0, 255, 0),
                (
                    segment[0] * self.block_size,
                    segment[1] * self.block_size,
                    self.block_size,
                    self.block_size,
                ),
            )

        pygame.draw.rect(
            screen,
            (255, 0, 0),
            (
                self.food[0] * self.block_size,
                self.food[1] * self.block_size,
                self.block_size,
                self.block_size,
            ),
        )

        # Display time in top left
        elapsed_time = int(time.time() - self.start_time)
        time_text = font.render(f"Time: {elapsed_time}s", True, (255, 255, 255))
        screen.blit(time_text, (10, 10))

        # Display score in top right
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        score_rect = score_text.get_rect()
        score_rect.topright = (self.window_size - 10, 10)
        screen.blit(score_text, score_rect)

        pygame.display.flip()
        clock.tick(10)

    def close(self) -> None:
        if self.render_mode == "human":
            pygame.quit()
