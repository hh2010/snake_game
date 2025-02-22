import pickle
from snake_env import ImprovedSnakeEnv

def get_best_action(Q_table, state, actions):
    if state not in Q_table:
        return random.choice(actions)
    return max(Q_table[state], key=Q_table[state].get)

def play_snake():
    with open("./models/q_table.pkl", "rb") as f:
        Q_table = pickle.load(f)

    env = ImprovedSnakeEnv(grid_size=10)
    state = env.reset()

    total_reward = 0
    while True:
        action = get_best_action(Q_table, state, ['UP', 'DOWN', 'LEFT', 'RIGHT'])
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break
    
    print(f"Game Over! Final Score: {total_reward}")

play_snake()
