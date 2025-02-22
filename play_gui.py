import pickle
import pygame
from snake_env import ImprovedSnakeEnv

def get_best_action(Q_table, state, actions):
    if state not in Q_table:
        return random.choice(actions)
    return max(Q_table[state], key=Q_table[state].get)

def play_snake_gui():
    with open("q_table.pkl", "rb") as f:
        Q_table = pickle.load(f)

    env = ImprovedSnakeEnv(grid_size=10, block_size=40, render_mode="human")
    state = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = get_best_action(Q_table, state, ['UP', 'DOWN', 'LEFT', 'RIGHT'])
        next_state, _, done = env.step(action)
        state = next_state

        env.render()

        if done:
            running = False

    env.close()
    print("Game Over!")

play_snake_gui()
