import numpy as np
import random
import tensorflow as tf
from collections import deque

class ReplayBuffer:
    def __init__(s, buffer_size):
        s.buffer = deque(maxlen=buffer_size)

    def add(s, experience):
        s.buffer.append(experience)

    def sample(s, batch_size):
        batch = random.sample(s.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

class QNetwork(tf.keras.Model):
    def __init__(s, state_size, action_size, hidden_size=64):
        super(QNetwork, s).__init__()
        s.fc1 = tf.keras.layers.Dense(hidden_size, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), activation='relu', input_shape=(state_size,))
        s.fc2 = tf.keras.layers.Dense(hidden_size, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), activation='relu')
        s.fc3 = tf.keras.layers.Dense(action_size, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01))

    def call(s, inputs):
        x = s.fc1(inputs)
        x = s.fc2(x)
        x = s.fc3(x)
        return x



class DDQNAgent:
    def __init__(s, state_size, action_size, buffer_size=10000, batch_size=64, gamma=0.99, lr=0.001, update_every=100):
        s.state_size = state_size
        s.action_size = action_size
        s.memory = ReplayBuffer(buffer_size)
        s.batch_size = batch_size
        s.gamma = gamma
        s.q_network = QNetwork(state_size, action_size)
        s.target_network = QNetwork(state_size, action_size)
        s.optimizer = tf.keras.optimizers.Adam(lr)
        s.update_every = update_every
        s.step_count = 0

    def remember(s, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        s.memory.add(experience)

    def epsilon_greedy_policy(s, state, epsilon):
        state = np.expand_dims(state, axis=0)
        q_values = s.q_network.predict(state)
        if np.random.rand() < epsilon:
            return np.random.choice(s.action_size)
        else:
            return np.argmax(q_values[0])
        
    def act(s, state):
        state = np.expand_dims(state, axis=0)
        q_values = s.q_network.predict(state)
        return np.argmax(q_values[0])

    def update_target_network(s):
        s.target_network.set_weights(s.q_network.get_weights())

    def train(s):
        if len(s.memory.buffer) < s.batch_size:
            return

        states, actions, rewards, next_states, dones = s.memory.sample(s.batch_size)

        target_q_values_next = s.target_network.predict(next_states)
        max_actions = np.argmax(s.q_network.predict(next_states), axis=1)
        selected_target_q_values_next = target_q_values_next[np.arange(s.batch_size), max_actions]
        target_q_values = rewards + (1 - dones) * s.gamma * selected_target_q_values_next

        with tf.GradientTape() as tape:
            q_values = s.q_network(states, training=True)
            action_masks = tf.one_hot(actions, s.action_size)
            selected_q_values = tf.reduce_sum(action_masks * q_values, axis=1)
            loss = tf.reduce_mean(tf.square(target_q_values - selected_q_values))

        gradients = tape.gradient(loss, s.q_network.trainable_variables)
        s.optimizer.apply_gradients(zip(gradients, s.q_network.trainable_variables))

        s.step_count += 1
        if s.step_count % s.update_every == 0:
            s.update_target_network()
            
    def load_model(s, model_file):
            s.q_network.load_weights(model_file)
            s.update_target_network()
