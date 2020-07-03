import gym
import highway_env
from agent import Agent
import pandas as pd
import numpy as np

env = gym.make("highway-v0")

done = False

# Notes
# Action space between 0 and 4 inclusive
# 0 is merge left
# 1 is do nothing
# 2 is merge right
# 3 is speed up
# 4 is slow down
#
## Obs space is a 5x5 matrix with values between -1 and 1
## This represents a matrix with the labels:
##     presence, x, y, vx, vy: Ego Vehicle
##     presence, x, y, vx, vy: VEHICLE 1
##     presence, x, y, vx, vy: VEHICLE 2
##     presence, x, y, vx, vy: VEHICLE 3

while not done:
    obs, _, _, _ = env.step(int(input()))

    # print(pd.DataFrame.from_records([env.vehicle.to_dict()])["x", "y", "vx", "vy"])
    ego_dict = env.vehicle.to_dict()
    ego_agent = Agent(
        np.array([ego_dict["x"], ego_dict["y"]]),
        np.array([10000, ego_dict["y"]]),
        50,
        55,
        1,
    )
    neighbors = []
    for vehicle in env.road.close_vehicles_to(
        env.vehicle, env.PERCEPTION_DISTANCE, see_behind=False
    ):
        adj_dict = env.vehicle.to_dict()
        neighbors.append(
            Agent(
                np.array([adj_dict["x"], adj_dict["y"]]),
                np.array([10000, adj_dict["y"]]),
                50,
                55,
                1,
            )
        )

    print(ego_agent.computeForces(neighbors))

    print(env.vehicle.to_dict())

    env.render()
