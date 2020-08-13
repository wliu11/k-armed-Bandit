#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:06:06 2020

@author: liuwendy
"""

import numpy as np


num_steps = 10000
num_episodes = 300


class Bandit:
    
    def __init__(self, k=10, epsilon=0.1):
        # Set epsilon value to the given value of 0.1
        self.epsilon = epsilon
        
        # Set the number of arms
        self.k = k
        
        # Initialize q(a), Q(a) and N(a) both to 0
        self.q = np.zeros(self.k, dtype=np.float)   # True values
        self.Q_sample = np.zeros(self.k, dtype=np.float)   # Estimated values
        self.Q = np.zeros(self.k, dtype=np.float)   # Estimated values

        self.N = np.zeros(self.k, dtype=np.int)   # Number of actions
        
        
    def update(self, action, reward, sample_avg, step_size=0.1):
        self.N[action] += 1

        # Update true value for each arm with noise
        noise = np.random.normal(0, 0.01, self.k)
        self.q = self.q + noise
        
        # Action-value method using sample averages
        if sample_avg:
            self.Q_sample[action] += (1.0 / self.N[action]) * (reward - self.Q_sample[action])
        
        # Action-value method using a constant step-size parameter
        else:
            self.Q[action] += step_size * (reward - self.Q[action])

        
    def reward(self, action):
        return np.random.normal(self.q[action], 1)
    
      
    # np.random.random get the probability
    def action(self, sample_avg):
        is_optimal = 1;        
        optimal_arm =  np.argmax(self.q)
                                 
        # Explore
        if np.random.random() < self.epsilon:
            random_arm = np.random.randint(self.k)            
            if (random_arm != optimal_arm):
                is_optimal = 0
            return (random_arm, is_optimal)
        
        # Exploit
        if sample_avg:
            presumed_optimal = np.random.choice(np.flatnonzero(self.Q_sample == self.Q_sample.max()))
            
        else:
            presumed_optimal = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
        
        if (presumed_optimal != optimal_arm):
            is_optimal = 0
        
        return (presumed_optimal, is_optimal)


def run_experiment(num_episodes, num_steps):
    
    results = np.zeros((4, num_steps), float)

    # Sample average bandit
    for i in range(num_episodes):
        bandit = Bandit()
        
        for j in range(num_steps):
            action, is_optimal = bandit.action(sample_avg=True)
            reward = bandit.reward(action)
            bandit.update(action, reward, sample_avg=True)
            
            results[0, j] += reward
            results[1, j] += is_optimal
            
            # Non sample average bandit
            action, is_optimal = bandit.action(sample_avg=False)
            reward = bandit.reward(action)
            bandit.update(action, reward, sample_avg=False)    
            
            results[2, j] += reward 
            results[3, j] += is_optimal
            
    results /= num_episodes
    return results
 


np.savetxt('result.out', run_experiment(num_episodes, num_steps))

  
