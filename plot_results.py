import matplotlib.pyplot as plt
import pickle

# Load training rewards
with open("./models/training_rewards.pkl", "rb") as f:
    episode_rewards = pickle.load(f)

# Plot results
plt.plot(episode_rewards, label="Training Rewards")
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress of Q-Learning Snake Agent')
plt.legend()
plt.show()
