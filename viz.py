import pygame
import torch
import numpy as np
from env import Environment
from ppo import PPO

pygame.init()
env = Environment()

env.screen = pygame.display.set_mode((env.screen_width, env.screen_height))
env.font = pygame.font.SysFont("Arial", 18)
clock = pygame.time.Clock()

# env.max_runner_speed = 5
# env.max_chaser_speed = 10
env.friction = 0.2
max_accel = env.chaser_acceleration

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

state_dim = 15
action_dim = 2
ppo_agent = PPO(
    state_dim=state_dim,
    action_dim=action_dim,
    lr_actor=3e-4,
    lr_critic=3e-4,
    gamma=0.99,
    K_epochs=10,
    eps_clip=0.2,
    has_continuous_action_space=True
)

ppo_agent.policy_old.load_state_dict(torch.load(
    "ppo_weights/attempt-2/ppo_network_ep9000.pth",
    map_location=device
))
ppo_agent.policy_old.to(device)
ppo_agent.policy_old.eval()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state, _ = env.update()
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

    s_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
    action_raw, _, _ = ppo_agent.policy_old.act(s_tensor)
    action_raw = action_raw.detach().cpu().numpy().flatten()
    action_scaled = action_raw * max_accel

    env.action_continuous(action_scaled)
    env.render()
    clock.tick(60)

pygame.quit()
