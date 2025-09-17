import pygame
import random
import math


class Environment:
    def __init__(self):
        # screen and game
        self.screen_width, self.screen_height = 1240, 800
        self.border_width = 10
        self.score = 0
        self.friction = 0.7

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.font = pygame.font.SysFont("Arial", 18)
        pygame.display.set_caption("Environment")

        # chaser
        self.chaser_size = 10
        self.chaser_color = (0, 0, 255)
        self.chaser_acceleration = 0.8
        self.chaser_pos = []
        self.chaser_vel = []
        self.max_chaser_speed = 25

        # runner
        self.runner_size = 20
        self.runner_color = (255, 0, 0)
        self.runner_level_acceleration = 2.0
        self.runner_speed = 0
        self.runner_pos = []
        self.runner_vel = []
        self.max_runner_speed = 20
        self.runner_step_counter = 0
        self.runner_step_threshold = 50

        self.prev_distance = 0

        self.reset()

    def reset(self):
        # init position
        self.chaser_pos = [float(self.screen_width // 2), float(self.screen_height // 2)]
        self.runner_pos = [
            random.randint(self.border_width, self.screen_width - self.border_width),
            random.randint(self.border_width, self.screen_height - self.border_width)
        ]

        # init speed
        self.chaser_vel = [0.0, 0.0]
        self.runner_speed = 5

        # init runner step counter
        self.runner_step_counter = 0

        self.prev_distance = ((self.chaser_pos[0] - self.runner_pos[0])**2 +
                              (self.chaser_pos[1] - self.runner_pos[1])**2)**0.5

        angle = random.uniform(0, 2 * math.pi)
        self.runner_vel = [self.runner_speed * math.cos(angle), self.runner_speed * math.sin(angle)]

        # init score
        self.score = 0

    def render(self):
        self.screen.fill((255, 255, 255))

        pygame.draw.rect(self.screen, (0, 0, 0),
                         (0, 0, self.screen_width, self.screen_height), self.border_width)

        pygame.draw.rect(self.screen, self.chaser_color,
                         (int(self.chaser_pos[0] - self.chaser_size / 2),
                          int(self.chaser_pos[1] - self.chaser_size / 2),
                          self.chaser_size, self.chaser_size))

        pygame.draw.rect(self.screen, self.runner_color,
                         (int(self.runner_pos[0] - self.runner_size / 2),
                          int(self.runner_pos[1] - self.runner_size / 2),
                          self.runner_size, self.runner_size))

        score_text = f"Score: {self.score}"
        chaser_speed_text = f"Chaser Speed: {math.sqrt(self.chaser_vel[0]**2 + self.chaser_vel[1]**2):.2f}"
        runner_speed_text = f"Runner Speed: {math.sqrt(self.runner_vel[0]**2 + self.runner_vel[1]**2):.2f}"

        text_surface1 = self.font.render(score_text, True, (0, 0, 0))
        text_surface2 = self.font.render(chaser_speed_text, True, (0, 0, 0))
        text_surface3 = self.font.render(runner_speed_text, True, (0, 0, 0))

        screen_offset = self.border_width + 5  # padding
        self.screen.blit(text_surface1, (20, screen_offset))
        self.screen.blit(text_surface2, (200, screen_offset))
        self.screen.blit(text_surface3, (500, screen_offset))

        pygame.display.flip()

    def handle_input(self):
        keys = pygame.key.get_pressed()
        action = None

        if keys[pygame.K_UP]:
            action = 0
        elif keys[pygame.K_DOWN]:
            action = 1
        elif keys[pygame.K_LEFT]:
            action = 2
        elif keys[pygame.K_RIGHT]:
            action = 3

        if action is not None:
            self.action(action)

    def action(self, action):
        ax, ay = 0.0, 0.0
        if action == 0:
            ay = -self.chaser_acceleration
        elif action == 1:
            ay = self.chaser_acceleration
        elif action == 2:
            ax = -self.chaser_acceleration
        elif action == 3:
            ax = self.chaser_acceleration

        self.chaser_vel[0] += ax
        self.chaser_vel[1] += ay

        if ax == 0:
            self.chaser_vel[0] *= self.friction
        if ay == 0:
            self.chaser_vel[1] *= self.friction

        self.chaser_vel[0] = max(-self.max_chaser_speed, min(self.chaser_vel[0], self.max_chaser_speed))
        self.chaser_vel[1] = max(-self.max_chaser_speed, min(self.chaser_vel[1], self.max_chaser_speed))

    def update(self):
        reward = 0
        wall_hit = False

        self.runner_pos[0] += self.runner_vel[0]
        self.runner_pos[1] += self.runner_vel[1]
        self.chaser_pos[0] += self.chaser_vel[0]
        self.chaser_pos[1] += self.chaser_vel[1]

        if self.runner_pos[0] <= self.border_width or self.runner_pos[0] >= self.screen_width - self.border_width:
            self.runner_vel[0] *= -1
            self.runner_pos[0] = max(self.border_width, min(self.runner_pos[0], self.screen_width - self.border_width))
            reward = -5
            wall_hit = True

        if self.runner_pos[1] <= self.border_width or self.runner_pos[1] >= self.screen_height - self.border_width:
            self.runner_vel[1] *= -1
            self.runner_pos[1] = max(self.border_width, min(self.runner_pos[1], self.screen_height - self.border_width))
            reward = -5
            wall_hit = True

        if (self.chaser_pos[0] <= self.border_width or self.chaser_pos[0] >= self.screen_width - self.border_width
                or self.chaser_pos[1] <= self.border_width or self.chaser_pos[1] >= self.screen_height - self.border_width):
            self.chaser_pos = [self.screen_width / 2, self.screen_height / 2]
            self.chaser_vel = [0.0, 0.0]

        self.runner_step_counter += 1
        if self.runner_step_counter >= self.runner_step_threshold:
            runner_speed_magnitude = (self.runner_vel[0]**2 + self.runner_vel[1]**2)**0.5
            if runner_speed_magnitude != 0:
                angle = random.uniform(0, 2 * math.pi)
                self.runner_vel[0] = runner_speed_magnitude * math.cos(angle)
                self.runner_vel[1] = runner_speed_magnitude * math.sin(angle)
            self.runner_step_counter = 0

        dx = self.chaser_pos[0] - self.runner_pos[0]
        dy = self.chaser_pos[1] - self.runner_pos[1]
        distance = (dx**2 + dy**2)**0.5

        if not wall_hit:
            if distance < self.prev_distance:
                reward += 1

        if distance <= self.chaser_size + self.runner_size:
            reward = 10
            self.score += 1

            self.runner_pos = [
                random.randint(self.border_width, self.screen_width - self.border_width),
                random.randint(self.border_width, self.screen_height - self.border_width)
            ]

            speed_magnitude = (self.runner_vel[0]**2 + self.runner_vel[1]**2)**0.5
            if speed_magnitude != 0:
                scale = (speed_magnitude + self.runner_level_acceleration) / speed_magnitude
                self.runner_vel[0] *= scale
                self.runner_vel[1] *= scale

                final_speed = (self.runner_vel[0]**2 + self.runner_vel[1]**2)**0.5
                if final_speed > self.max_runner_speed:
                    factor = self.max_runner_speed / final_speed
                    self.runner_vel[0] *= factor
                    self.runner_vel[1] *= factor

            self.runner_step_counter = 0

        self.prev_distance = distance

        state = {
            "runner_pos": self.runner_pos[:],
            "chaser_pos": self.chaser_pos[:],
            "chaser_speed": (self.chaser_vel[0]**2 + self.chaser_vel[1]**2)**0.5
        }

        return state, reward
