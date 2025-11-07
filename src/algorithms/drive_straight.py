#!/usr/bin/env python3
import os
import gym
import gym_donkeycar
import numpy as np

import util


# SET UP ENVIRONMENT
# You can also launch the simulator separately
# in that case, you don't need to pass a `conf` object
port = 9091

conf = { "exe_path" : util.find_sim_per_platform(), "port" : port }

#env = gym.make("donkey-circuit-launch-track-v0", conf=conf)
env = gym.make("donkey-circuit-launch-track-v0")

# PLAY
obs = env.reset()
for t in range(100):
  action = np.array([0.0, 0.5]) # drive straight with small speed
  # execute the action
  obs, reward, done, info = env.step(action)

# Exit the scene
env.close()
