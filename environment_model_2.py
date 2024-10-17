import pygame
import random
import math
import time
import threading
from dqn import *

class Environment:
    def __init__(s):
        s.screen_width = 1240
        s.screen_height = 800
        s.border_width = 10
        s.player_size = 20
        s.goal_size = 40
        s.player_color = (0, 0, 255)
        s.goal_color = (255, 0, 0)
        s.player_speed = 8
        s.goal_speed = 1
        s.constant_goal_speed_increment = 0.002
        s.level_up_goal_speed_increment = 0.2
        s.player_acceleration = 0.6
        s.max_player_speed = 75
        s.goal_change_threshold = 200
        
        s.player_positions = [s.screen_width//2, s.screen_height//2]
        
        s.reset()

    def reset(s):
        s.player_x = s.screen_width // 2
        s.player_y = s.screen_height // 2
        
        s.goal_x = random.randint(s.border_width + 375, s.screen_width - s.border_width - s.goal_size - 375)
        s.goal_y = random.randint(s.border_width + 215, s.screen_height - s.border_width - s.goal_size - 215)
        
        s.goal_dx = random.choice([-1, 1]) * s.goal_speed
        s.goal_dy = random.choice([-1, 1]) * s.goal_speed
        s.reward = 0
        s.steps = 0
        
        s.player_positions = [s.screen_width//2, s.screen_height//2]
        
        return s.getState()
        

    def playerAction(s, action):
        # if action == 0:
        #     pass
        if action == 0:
            s.player_y -= s.player_speed
        elif action == 1:
            s.player_y += s.player_speed
        elif action == 2:
            s.player_x -= s.player_speed
        elif action == 3:
            s.player_x += s.player_speed
        elif action == 4:
            s.player_speed = min(s.player_speed + s.player_acceleration, s.max_player_speed)
        elif action == 5:
            s.player_speed = max(8, s.player_speed - s.player_acceleration)
        
        s.player_x = max(s.border_width, min(s.player_x, s.screen_width - s.border_width - s.player_size))
        s.player_y = max(s.border_width, min(s.player_y, s.screen_height - s.border_width - s.player_size))
            
        # time.sleep(0.001)
            

    def goalMovement(s):
        s.goal_x += s.goal_dx
        s.goal_y += s.goal_dy
        
        if (s.goal_x <= s.border_width  + 10 or s.goal_x >= s.screen_width - s.border_width - s.goal_size - 10) and s.steps % 50 == 0:
            s.goal_dx *= -1
        if (s.goal_y <= s.border_width  + 10 or s.goal_y >= s.screen_height - s.border_width - s.goal_size - 10) and s.steps % 50 == 0:
            s.goal_dy *= -1
            

        # Check if the goal hits the border
        s.goal_x = max(s.border_width, min(s.goal_x, s.screen_width - s.border_width - s.goal_size))
        s.goal_y = max(s.border_width, min(s.goal_y, s.screen_height - s.border_width - s.goal_size))

        s.steps += 1
        
        # constant goal speed increment
        # s.goal_speed += s.constant_goal_speed_increment

        # change directions in certain interval of time
        if s.steps % s.goal_change_threshold == 0:
            s.goal_dx = random.choice([-1, 1]) * s.goal_speed
            s.goal_dy = random.choice([-1, 1]) * s.goal_speed


    def getDistance(s):
        distance = math.sqrt((s.goal_x - s.player_x)**2 + (s.goal_y - s.player_y)**2)
        return distance

    def getReward(s):
        current_distance = s.getDistance()
           
        if current_distance <= s.goal_size:
            s.reward += 10
            return 10
        elif (s.player_x <= s.border_width or s.player_x >= s.screen_width - s.border_width - s.player_size or
              s.player_y <= s.border_width or s.player_y >= s.screen_height - s.border_width - s.player_size):
            s.reward -= 10
            
            s.player_x = s.screen_width//2
            s.player_y = s.screen_height//2
            
            return -10
        else:
            if len(s.player_positions) == 2 and type(s.player_positions[0]) == list:
                prev_distance = math.sqrt((s.goal_x - s.player_positions[0][0])**2 + (s.goal_y - s.player_positions[0][1])**2)
                if current_distance < prev_distance:
                    return 1
                else:
                    return -2  
            else:
                return 0

    def step(s, action):
        s.playerAction(action)
        s.goalMovement()
        
        s.player_positions.append([s.player_x, s.player_y])
        if len(s.player_positions) > 2:
            s.player_positions.pop(0)
        
        reward = s.getReward()
        state = s.getState()
        
        done = False
        
        if reward == 10 or reward == -10:
            done = True
            s.steps = 0
            s.reward = 0
            if reward == 10:   # player caught the goal
        
                s.goal_x = random.randint(s.border_width + 375, s.screen_width - s.border_width - s.goal_size - 375)
                s.goal_y = random.randint(s.border_width + 215, s.screen_height - s.border_width - s.goal_size - 215)   
                
                s.goal_dx = random.choice([-1, 1]) * s.goal_speed
                s.goal_dy = random.choice([-1, 1]) * s.goal_speed
                
                s.goal_speed = min(s.goal_speed + s.level_up_goal_speed_increment, 50)   # goal speed increase when it levels up
        return state, reward, done


    def getState(s):
        distance = s.getDistance()
        return s.goal_x, s.goal_y, s.player_x, s.player_y, distance, s.player_speed, s.goal_speed, s.screen_height, s.screen_width, s.border_width
    
    def render(s):
        pygame.init()
        screen = pygame.display.set_mode((s.screen_width, s.screen_height))
        pygame.display.set_caption("RL Environment")
        
        s.running = True
        
        return screen
    
    def updateScreen(s, screen):
        
        screen.fill((255, 255, 255))
        pygame.draw.rect(screen, s.player_color, (s.player_x, s.player_y, s.player_size, s.player_size))
        pygame.draw.rect(screen, s.goal_color, (s.goal_x, s.goal_y, s.goal_size, s.goal_size))
        pygame.draw.rect(screen, (0, 0, 0), (0, 0, s.screen_width, s.screen_height), s.border_width)
        pygame.display.flip()
        
    def close_window(s):
        pygame.quit()
        
        
    def renderWithAgent(s, agent):
        pygame.init()
        screen = pygame.display.set_mode((s.screen_width, s.screen_height))
        pygame.display.set_caption("RL Environment")
        clock = pygame.time.Clock()
        state = s.reset()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # agent plays the game
            
            action = agent.act(state)
            print(action)
            next_state, _, _ = s.step(action)
            state = next_state

            screen.fill((255, 255, 255))
            pygame.draw.rect(screen, s.player_color, (s.player_x, s.player_y, s.player_size, s.player_size))
            pygame.draw.rect(screen, s.goal_color, (s.goal_x, s.goal_y, s.goal_size, s.goal_size))
            pygame.draw.rect(screen, (0, 0, 0), (0, 0, s.screen_width, s.screen_height), s.border_width)
            pygame.display.flip()
            
            clock.tick(60)
        pygame.quit()
        
        
if __name__ == "__main__":
    env = Environment()
    
    state_size = 10
    action_size = 6
    agent = DDQNAgent(state_size=state_size, action_size=action_size)
    
    # agent.q_network.load_weights('300_dynamic_level_up.weights.h5')
    
    num_episodes = 500
    max_steps_per_episode = 500
    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_decay = 0.991
    
    state = env.reset()
    screen = env.render()
    
    for episode in range(num_episodes):
        state = env.getState()
        
        total_reward = 0
        
        epsilon = max(epsilon_end, epsilon_start*epsilon_decay**episode)
        
        for step in range(max_steps_per_episode):
            
            action = agent.epsilon_greedy_policy(state, epsilon)
            
            next_state, reward, done = env.step(action)
            if reward == 10:
                print('win')
                
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            agent.train()

            if done:
                break
            
            env.updateScreen(screen)
            
            print(f"Current Episode: {episode + 1}, At step: {step + 1}")

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
        
        
    agent.q_network.save_weights("300_dynamic_level_up_plus.weights.h5")
    agent.q_network.save("300_dynamic_level_up_plus.keras")
    
    # env.renderWithAgent(agent)
