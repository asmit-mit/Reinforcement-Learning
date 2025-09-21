import pygame
from env import Environment

pygame.init()
env = Environment()

pygame.display.set_caption("Environment")
env = Environment()
env.screen = pygame.display.set_mode((env.screen_width, env.screen_height))
env.font = pygame.font.SysFont("Arial", 18)

clock = pygame.time.Clock()

# env.max_runner_speed = 5
# env.max_chaser_speed = 10
env.friction = 1

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    env.handle_input()
    state, reward = env.update()
    env.render()

    print("State:", state)
    print("Reward:", reward)

    clock.tick(60)

pygame.quit()
