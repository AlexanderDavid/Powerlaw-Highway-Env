#! /usr/bin/env python

import gym
import highway_env

env = gym.make("highway-v0")

done = False
while not done:
    action = env.action_space.sample() # Your agent code here
    obs, reward, done, _ = env.step(1)
    env.render()

env.close()
