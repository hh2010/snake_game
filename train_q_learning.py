import random
import pickle
from snake_env import ImprovedSnakeEnv
import matplotlib.pyplot as plt

# Define hyperparameters
alpha = 0.1   # Learning rate
gamma = 0.9   # Discount factor
epsilon = 1.0 # Exploration rate (starts high, decays over time)
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 5000
actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# Initialize Q-table as a dictionary
Q_table = {}

def get_best_action(state):
    """Selects the action with the highest Q-value for a given state."""
    if state not in Q_table:
        Q_table[state] = {a: 0 for a in actions}  # Initialize unseen states with 0
    
    return max(Q_table[state], key=Q_table[state].get)  # Action with highest Q-value

def train_q_learning():
    global epsilon
    env = ImprovedSnakeEnv(grid_size=10)
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)
            else:
                action = get_best_action(state)

            next_state, reward, done = env.step(action)
            total_reward += reward

            if state not in Q_table:
                Q_table[state] = {a: 0 for a in actions}
            if next_state not in Q_table:
                Q_table[next_state] = {a: 0 for a in actions}

            best_next_action = get_best_action(next_state)
            Q_table[state][action] = Q_table[state][action] + alpha * (reward + gamma * Q_table[next_state][best_next_action] - Q_table[state][action])

            state = next_state

            if done:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        episode_rewards.append(total_reward)

    with open("./models/q_table.pkl", "wb") as f:
        pickle.dump(Q_table, f)

  
  
# Train and plot results
episode_rewards = train_q_learning()

plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress of Q-Learning Snake Agent')
plt.savefig("./outputs/training_plot.png")
plt.show()
