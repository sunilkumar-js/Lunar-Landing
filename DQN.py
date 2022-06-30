
import torch
import numpy as np
import sys
import torch.nn.functional as F
from utils import *
import pygame


class DQNAgent():

    def __init__(self, env, lr, epsilon, epsilon_min, gamma, decay_factor):

        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.decay_factor = decay_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.model = DQN(env.observation_space.shape[0], env.action_space.n)
        # self.model_target = DQN(env.observation_space.shape[0],env.action_space.n)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.batch_size = 64
        self.replay_buffer = BasicBuffer(int(1e5))
        self.lossFn = F.mse_loss
        self.episodes_rewards = []
        self.target = 200
        self.update_size = 4
        self.counter = 0
        #self.tau = tau

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_factor)

    def update_model(self):
        batch = self.replay_buffer.sample(self.batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_loss(self, batch):

        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        curr_qval = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_qvals = self.model(next_states)
        max_next_action = torch.max(next_qvals, 1)[0]
        expected_qval = rewards.squeeze(1) + (1 - dones) * self.gamma * max_next_action

        loss = self.lossFn(curr_qval, expected_qval.detach())
        return loss

    def get_action(self, observation):
        observation = torch.FloatTensor(observation).float().unsqueeze(0)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_space.n)
        self.model.eval()
        with torch.no_grad():
            qvals = self.model(observation)
        self.model.train()
        return np.argmax(qvals.detach().numpy())

    def train(self, total_episodes=2000, early_stop=True):

        for episode in range(total_episodes):
            observation, info = self.env.reset(return_info=True)

            episode_reward = 0

            for t in range(1000):
                curr_action = self.get_action(observation)

                next_observation, reward, done, info = self.env.step(curr_action)

                self.counter += 1
                self.replay_buffer.push(observation, curr_action, reward, next_observation, done)

                if len(self.replay_buffer) > self.batch_size and self.counter % self.update_size == 0:
                    self.update_model()

                observation = next_observation
                episode_reward += reward
                if done:
                    break
            self.decay_epsilon()
            self.episodes_rewards.append(episode_reward)
            avg_reward = np.mean(self.episodes_rewards[-100:])
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, avg_reward), end="")
            if episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, avg_reward))
            if avg_reward > self.target and early_stop:
                print("Mission Accomplished")
                break
        self.env.close()

    def play(self):
        pygame.init()
        frames =[]
        observation = self.env.reset()
        for t in range(1000):
            frames.append(self.env.render(mode="rgb_array"))
            action = self.get_action(observation)
            observation,_,done,_ = self.env.step(action)
            if done:
                break
        self.env.close()
        return frames

