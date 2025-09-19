import pygame
import torch
import numpy as np
from env import Environment
from dqn import Agent

pygame.init()
env = Environment()

env.screen = pygame.display.set_mode((env.screen_width, env.screen_height))
env.font = pygame.font.SysFont("Arial", 18)
env.max_runner_speed = 5
env.max_chaser_speed = 10
env.friction = 0.5

clock = pygame.time.Clock()

agent = Agent(
    input_dim=13,
    output_dim=4,
    hidden_dim=128,
)

agent.q_net.load_state_dict(torch.load("weights/attempt-3/best_q_network_ep1802.pth", map_location=agent.device))
agent.q_net.eval()


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state, reward = env.update()
    print(reward)

    s = np.array([
        state["runner_pos"][0], state["runner_pos"][1],
        state["chaser_pos"][0], state["chaser_pos"][1],
        state["runner_vel"][0], state["runner_vel"][1],
        state["chaser_vel"][0], state["chaser_vel"][1],
        state["distances"][0], state["distances"][1],
        state["distances"][2], state["distances"][3],
        state["friction"]
    ], dtype=np.float32)

    action = agent.select_action(s, epsilon=0)
    env.action(action)

    env.render()
    clock.tick(60)

pygame.quit()
