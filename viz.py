import pygame
import torch
import numpy as np
from env import Environment
from dqn import Agent

pygame.init()
env = Environment()
clock = pygame.time.Clock()

agent = Agent(
    input_dim=8,
    output_dim=4,
    hidden_dim=64,
    buffer_size=10000,
    batch_size=32,
    gamma=0.99,
    lr=1e-3,
)

agent.q_net.load_state_dict(torch.load("weights/demo.pth", map_location=agent.device))
agent.q_net.eval()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state, _ = env.update()
    print(state)

    s = np.array([
        state["runner_pos"][0],
        state["runner_pos"][1],
        state["chaser_pos"][0],
        state["chaser_pos"][1],
        state["runner_vel"][0],
        state["runner_vel"][1],
        state["chaser_vel"][0],
        state["chaser_vel"][1],
    ], dtype=np.float32)

    action = agent.select_action(s, epsilon=0)  # no randomness, full control
    env.action(action)

    env.render()
    clock.tick(60)

pygame.quit()
