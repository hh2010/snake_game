import random
import pygame

class ImprovedSnakeEnv:
    def __init__(self, grid_size=10, block_size=40, render_mode="none"):
        self.grid_size = grid_size
        self.block_size = block_size
        self.render_mode = render_mode
        self.reset()

        if self.render_mode == "human":
            pygame.init()
            self.window_size = self.grid_size * self.block_size
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

    def reset(self):
        self.snake = [(random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))]
        self.food = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        self.direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        self.done = False
        return self.get_state()

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        food_left = 1 if food_x < head_x else 0
        food_right = 1 if food_x > head_x else 0
        food_up = 1 if food_y < head_y else 0
        food_down = 1 if food_y > head_y else 0

        danger_left = 1 if (head_x - 1 < 0 or (head_x - 1, head_y) in self.snake) else 0
        danger_right = 1 if (head_x + 1 >= self.grid_size or (head_x + 1, head_y) in self.snake) else 0
        danger_up = 1 if (head_y - 1 < 0 or (head_x, head_y - 1) in self.snake) else 0
        danger_down = 1 if (head_y + 1 >= self.grid_size or (head_x, head_y + 1) in self.snake) else 0

        return (food_left, food_right, food_up, food_down, danger_left, danger_right, danger_up, danger_down)

    def step(self, action):
        head_x, head_y = self.snake[0]

        if action == 'UP': new_head = (head_x, head_y - 1)
        if action == 'DOWN': new_head = (head_x, head_y + 1)
        if action == 'LEFT': new_head = (head_x - 1, head_y)
        if action == 'RIGHT': new_head = (head_x + 1, head_y)

        if (new_head[0] < 0 or new_head[0] >= self.grid_size or 
            new_head[1] < 0 or new_head[1] >= self.grid_size or 
            new_head in self.snake):
            self.done = True
            return self.get_state(), -10, self.done

        self.snake.insert(0, new_head)

        if new_head == self.food:
            reward = 10
            self.food = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        else:
            self.snake.pop()
            old_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = 1 if new_dist < old_dist else -1

        return self.get_state(), reward, self.done

    def render(self):
        if self.render_mode == "none":
            return

        self.screen.fill((0, 0, 0))

        for segment in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), 
                             (segment[0] * self.block_size, segment[1] * self.block_size, self.block_size, self.block_size))

        pygame.draw.rect(self.screen, (255, 0, 0), 
                         (self.food[0] * self.block_size, self.food[1] * self.block_size, self.block_size, self.block_size))

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
