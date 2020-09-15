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
##
## X increases over time
## Y = 0 in top line
## Y = 4 in next line
## Y = 8 in next lane
## Y = 12 in bottom lane

next_step = 1
while not env.vehicle.crashed:
    obs, _, _, _ = env.step(next_step)

    # print(pd.DataFrame.from_records([env.vehicle.to_dict()])["x", "y", "vx", "vy"])
    ego_dict = env.vehicle.to_dict()
    ego_agent = Agent(
        np.array([ego_dict["x"], ego_dict["y"] / 4]),
        np.array([ego_dict["x"] + 100, ego_dict["y"] / 4]),
        50,
        50,
        5,
        np.array([ego_dict["vx"], ego_dict["vy"] / 4]),
    )
    print(f"Ego (x, y): {ego_agent.pos[0], ego_agent.pos[1], ego_agent.vel[0], ego_agent.vel[1]}")
    # print(f"Ego (lane, lane_index): {env.vehicle.lane, env.vehicle.lane_index}")
    neighbors = []
    for vehicle in env.road.close_vehicles_to(
        env.vehicle, env.PERCEPTION_DISTANCE, see_behind=True
    ):
        adj_dict = vehicle.to_dict()
        neighbors.append(
            Agent(
                np.array([adj_dict["x"], adj_dict["y"] / 4]),
                np.array([adj_dict["x"] + 100, adj_dict["y"] / 4]),
                50,
                50,
                5,
                np.array([adj_dict["vx"], adj_dict["vy"] / 4]),
            )
        )
        print(f"Neighbor (x, y): {neighbors[-1].pos[0], neighbors[-1].pos[1], neighbors[-1].vel[0], neighbors[-1].vel[1], ego_agent.time_to_collision(neighbors[-1])}")

    # Add agents so the ego doesnt merge off of the edge of the lane
    neighbors.append(
        Agent(
            np.array([-1, ego_dict["y"] / 4]),
            np.array([adj_dict["x"] + 100, adj_dict["y"] / 4]),
            50,
            50,
            5,
            np.array([ego_dict["vx"], 0.5]),
        )
    )

    neighbors.append(
        Agent(
            np.array([5, ego_dict["y"] / 4]),
            np.array([adj_dict["x"] + 100, adj_dict["y"] / 4]),
            50,
            50,
            5,
            np.array([ego_dict["vx"], -0.5]),
        )
    )

    delta_v = ego_agent.computeForces(neighbors)
    print(delta_v)

    # If the X instruction is larger
    # If the X instruction is positive
    # if abs(delta_v[0]) == delta_v[0]:
    #     print("Speed up")
    # else:
    #     print("Slow down")
    lane_epsilon = 0.0125
    move_epsilon = 0.01

    def how_close(x):
        return abs(round(x) - x), round(x)

    laneness = how_close(ego_agent.pos[1])
    can_change = False
    if laneness[1] in [0, 1, 2, 3] and lane_epsilon > laneness[0]:
        can_change = True

    if can_change and abs(delta_v[1]) > move_epsilon:
        if abs(delta_v[1]) == delta_v[1]:
            print("Merge down")
            next_step = 2
        else:
            print("Merge up")
            next_step = 0
    else:
        if abs(delta_v[0]) == delta_v[0]:
            print("Speed up")
            next_step = 3
        else:
            print("Slow down")
            next_step = 4

    env.render()
