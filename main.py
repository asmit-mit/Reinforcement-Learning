import pygame
from env import Environment

pygame.init()
env = Environment()
clock = pygame.time.Clock()

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
