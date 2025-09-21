import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import pygame
from collections import deque
from env import Environment
from ppo import PPO

base_dir = "ppo_weights"
os.makedirs(base_dir, exist_ok=True)

existing_attempts = [d for d in os.listdir(base_dir) if d.startswith("attempt-")]
attempt_nums = [int(d.split("-")[1]) for d in existing_attempts if d.split("-")[1].isdigit()]
next_attempt = max(attempt_nums, default=0) + 1

attempt_dir = os.path.join(base_dir, f"attempt-{next_attempt}")
os.makedirs(attempt_dir, exist_ok=True)

pygame.init()
pygame.display.set_caption("Runner-Chaser")
clock = pygame.time.Clock()

env = Environment()
env.max_chaser_speed = 10
env.max_runner_speed = 5
env.friction = 1

state_dim = 15
action_dim = 2
max_accel = env.chaser_acceleration

ppo_agent = PPO(
    state_dim=state_dim,
    action_dim=action_dim,
    lr_actor=3e-4,
    lr_critic=3e-4,
    gamma=0.99,
    K_epochs=10,
    eps_clip=0.2,
    has_continuous_action_space=True,
    action_std_init=0.6
)

max_steps_per_episode = 1024
num_episodes = 50000
train_every_episode = True
save_interval = 500
render = False
threshold_score = 4
score_window = 30
total_score_window = deque(maxlen=score_window)
episode_rewards = []
global_step = 0

if render:
    env.screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    env.font = pygame.font.SysFont("Arial", 18)

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

        action_raw = ppo_agent.select_action(s)
        action_scaled = action_raw * max_accel

        env.action_continuous(action_scaled)
        next_state, reward = env.update()
        done = (reward == env.reward_wall) or (step >= max_steps_per_episode)

        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)

        total_reward += reward

        if render:
            env.render()
            clock.tick(60)

        state = next_state
        step += 1
        global_step += 1

    if train_every_episode:
        ppo_agent.update()

    episode_rewards.append(total_reward)
    total_score_window.append(env.score)

    if (episode + 1) % save_interval == 0:
        torch.save(ppo_agent.policy_old.state_dict(), f"{attempt_dir}/ppo_network_ep{episode + 1}.pth")
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
            torch.save(ppo_agent.policy_old.state_dict(), f"{attempt_dir}/best_ppo_network_ep{episode + 1}.pth")

    episode_time = time.time() - start_time
    avg_score_display = sum(total_score_window) / len(total_score_window) if total_score_window else 0
    print(f"Episode {episode + 1}/{num_episodes} | Reward: {total_reward:.2f} | Score: {env.score} | Avg Score (last 30): {avg_score_display:.2f} | Time: {episode_time:.2f}s")

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
plt.savefig("ppo_rewards_plot.png")
print("Saved reward plot as ppo_rewards_plot.png")
