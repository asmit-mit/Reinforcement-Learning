from dqn import Agent
from env import Environment
import numpy as np
import pygame
import matplotlib.pyplot as plt
import torch


pygame.init()
pygame.display.set_caption("Environment")
env = Environment()
env.screen = pygame.display.set_mode((env.screen_width, env.screen_height))
env.font = pygame.font.SysFont("Arial", 18)

clock = pygame.time.Clock()

agent = Agent(
    input_dim=12,
    output_dim=4,
    hidden_dim=64,
    buffer_size=10000,
    batch_size=32,
    gamma=0.99,
    lr=1e-3,
)

max_steps_per_episode = 1000

epsilon = 1.0
epsilon_decay = 0.98
epsilon_end = 0.05
num_episodes = 1000

max_reward = 0
episode_rewards = []

env.max_chaser_speed = 10
env.max_runner_speed = 5
env.friction = 0

for episode in range(num_episodes):
    env.reset()
    state, _ = env.update()
    done = False
    total_reward = 0
    step = 0

    while not done and step < max_steps_per_episode:
        s = np.array([
            state["runner_pos"][0],
            state["runner_pos"][1],
            state["chaser_pos"][0],
            state["chaser_pos"][1],
            state["runner_vel"][0],
            state["runner_vel"][1],
            state["chaser_vel"][0],
            state["chaser_vel"][1],
            state["distances"][0],
            state["distances"][1],
            state["distances"][2],
            state["distances"][3],
        ], dtype=np.float32)

        action = agent.select_action(s, epsilon)

        env.handle_input()
        # env.action(action)

        next_state, reward = env.update()
        env.render()
        total_reward += reward

        ns = np.array([
            next_state["runner_pos"][0],
            next_state["runner_pos"][1],
            next_state["chaser_pos"][0],
            next_state["chaser_pos"][1],
            next_state["runner_vel"][0],
            next_state["runner_vel"][1],
            next_state["chaser_vel"][0],
            next_state["chaser_vel"][1],
            next_state["distances"][0],
            next_state["distances"][1],
            next_state["distances"][2],
            next_state["distances"][3],
        ], dtype=np.float32)

        done = (step >= max_steps_per_episode)

        agent.store_transition(s, action, reward, ns, done)
        agent.train_step()

        state = next_state
        step += 1

        clock.tick(30)

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward} - Epsilon: {epsilon:.3f} - Score: {env.score}")
    if total_reward > max_reward:
        max_reward = total_reward
        print("New Highest")

    epsilon = max(epsilon_end, epsilon * epsilon_decay)

pygame.quit()

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Episode")
plt.legend()
plt.grid()
plt.savefig("rewards_plot.png")
print("Reward plot saved as rewards_plot.png")

torch.save(agent.q_net.state_dict(), "weights/q_network.pth")
