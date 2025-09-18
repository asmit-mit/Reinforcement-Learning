import pygame
import numpy as np
import matplotlib.pyplot as plt
import torch
from env import Environment
from dqn import Agent

pygame.init()
pygame.display.set_caption("Runner-Chaser")
clock = pygame.time.Clock()

env = Environment()

agent = Agent(
    input_dim=12,
    output_dim=4,
    hidden_dim=128,
    buffer_size=10000,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    update_every=20
)

max_steps_per_episode = 1000
num_episodes = 20000

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_episodes = 500

train_every = 4
warmup_steps = 2000
save_interval = 1000
render = True

if render:
    env.screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    env.font = pygame.font.SysFont("Arial", 18)

# env.max_chaser_speed = 10
# env.max_runner_speed = 5
env.friction = 1

episode_rewards = []
max_reward = -float("inf")
epsilon = epsilon_start
global_step = 0

for episode in range(num_episodes):
    env.reset()
    state, _ = env.update()
    total_reward = 0
    step = 0
    done = False

    while not done and step < max_steps_per_episode:
        s = np.array([
            state["runner_pos"][0], state["runner_pos"][1],
            state["chaser_pos"][0], state["chaser_pos"][1],
            state["runner_vel"][0], state["runner_vel"][1],
            state["chaser_vel"][0], state["chaser_vel"][1],
            state["distances"][0], state["distances"][1],
            state["distances"][2], state["distances"][3],
        ], dtype=np.float32)

        action = agent.select_action(s, epsilon)
        env.action(action)
        next_state, reward = env.update()

        total_reward += reward

        ns = np.array([
            next_state["runner_pos"][0], next_state["runner_pos"][1],
            next_state["chaser_pos"][0], next_state["chaser_pos"][1],
            next_state["runner_vel"][0], next_state["runner_vel"][1],
            next_state["chaser_vel"][0], next_state["chaser_vel"][1],
            next_state["distances"][0], next_state["distances"][1],
            next_state["distances"][2], next_state["distances"][3],
        ], dtype=np.float32)

        done = (reward == -100) or (step >= max_steps_per_episode)

        agent.store_transition(s, action, reward, ns, done)

        if len(agent.replay_buffer) > warmup_steps and global_step % train_every == 0:
            agent.train_step()

        if render:
            env.render()

        state = next_state
        step += 1
        global_step += 1

    episode_rewards.append(total_reward)

    if total_reward > max_reward:
        max_reward = total_reward
        torch.save(agent.q_net.state_dict(), "weights/best_q_network.pth")
        print(f"New High Score! Saved model at episode {episode + 1}")

    if (episode + 1) % save_interval == 0:
        torch.save(agent.q_net.state_dict(), f"weights/q_network_ep{episode + 1}.pth")
        print(f"Model saved at episode {episode + 1}")

    epsilon = max(epsilon_end, epsilon_start - (episode / epsilon_decay_episodes) * (epsilon_start - epsilon_end))

    print(f"Episode {episode + 1}/{num_episodes} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f} | Score: {env.score}")

pygame.quit()

window = 20
moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Episode Rewards", alpha=0.5)
plt.plot(range(window - 1, num_episodes), moving_avg, label=f"{window}-Episode Moving Avg", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Over Time")
plt.legend()
plt.grid()
plt.savefig("rewards_plot.png")
print("Saved reward plot as rewards_plot.png")
