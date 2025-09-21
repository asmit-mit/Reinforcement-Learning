import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import pygame
from collections import deque
from env import Environment
from dqn import Agent


# making a save space
base_dir = "dqn_weights"
os.makedirs(base_dir, exist_ok=True)

existing_attempts = [d for d in os.listdir(base_dir) if d.startswith("attempt-")]
attempt_nums = [int(d.split("-")[1]) for d in existing_attempts if d.split("-")[1].isdigit()]
next_attempt = max(attempt_nums, default=0) + 1

attempt_dir = os.path.join(base_dir, f"attempt-{next_attempt}")
os.makedirs(attempt_dir, exist_ok=True)

# init environment
pygame.init()
pygame.display.set_caption("Runner-Chaser")
clock = pygame.time.Clock()

env = Environment()

# init agent and hyperparameters
agent = Agent(
    input_dim=15,
    output_dim=4,
    hidden_dim=128,
    buffer_size=2000,
    batch_size=128,
    gamma=0.99,
    lr=1e-3,
    update_every=100
)

# hyperparameters
max_steps_per_episode = 1000  # 1000
num_episodes = 80_000  # 20000

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_episodes = 35_000  # 2000

train_every = 4
warmup_steps = 1000
save_interval = 500
render = False

threshold_score = 4
score_window = 30
total_score_window = deque(maxlen=score_window)

if render:
    env.screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    env.font = pygame.font.SysFont("Arial", 18)

# initial environment
env.max_chaser_speed = 10
env.max_runner_speed = 5
env.friction = 1

episode_rewards = []
max_reward = -float("inf")
epsilon = epsilon_start
global_step = 0

# training loop
for episode in range(num_episodes):
    start_time = time.time()

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
            state["max_runner_speed"], state["max_chaser_speed"],
            state["friction"]
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
            next_state["max_runner_speed"], next_state["max_chaser_speed"],
            next_state["friction"]
        ], dtype=np.float32)

        done = (reward == env.reward_wall) or (step >= max_steps_per_episode)

        agent.store_transition(s, action, reward, ns, done)

        if len(agent.replay_buffer) > warmup_steps and global_step % train_every == 0:
            agent.train_step()

        if render:
            env.render()
            clock.tick(60)

        state = next_state
        step += 1
        global_step += 1

    episode_rewards.append(total_reward)
    total_score_window.append(env.score)

    episode_time = time.time() - start_time

    if (episode + 1) % save_interval == 0:
        torch.save(agent.q_net.state_dict(), f"{attempt_dir}/q_network_ep{episode + 1}.pth")
        print(f"Model saved at episode {episode + 1}")

    if len(total_score_window) == score_window:
        avg_score = sum(total_score_window) / score_window
        if avg_score >= threshold_score:
            total_score_window.clear()

            env.friction = max(0.2, env.friction - 0.05)
            env.max_chaser_speed = min(30, env.max_chaser_speed + 0.5)
            env.max_runner_speed = min(20, env.max_runner_speed + 0.5)

            print(
                f"Score threshold crossed → difficulty up!\n"
                f"    Max chaser speed: {env.max_chaser_speed} (+0.5)\n"
                f"    Max runner speed: {env.max_runner_speed} (+0.5)\n"
                f"    Friction: {env.friction} (-0.05)\n"
            )
            torch.save(agent.q_net.state_dict(), f"{attempt_dir}/best_q_network_ep{episode + 1}.pth")

    epsilon = max(epsilon_end, epsilon_start - (episode / epsilon_decay_episodes) * (epsilon_start - epsilon_end))

    print(f"Episode {episode + 1}/{num_episodes} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f} | Score: {env.score} | Avg Score (last 30): {sum(total_score_window) / score_window:.2f} | Time: {episode_time:.2f}s")

pygame.quit()

window = 500
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
