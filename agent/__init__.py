# agent.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the author.
#
# Author: Ioannis Karamouzas (ioannis@g.clemson.edu)

from math import sqrt
from typing import List, Any

import numpy as np


class Agent:
    def __init__(
        self,
        pos: np.array,
        goal: np.array,
        pref_speed: float,
        max_speed: float,
        radius: float,
        vel: np.array = np.zeros(2),
        ksi: float = 10,
        dist_hor: float = 50,
        time_hor: float = 50,
        max_f: float = 0,
    ):
        self.pos = pos
        self.vel = vel
        self.goal = goal
        self.prefspeed = pref_speed
        self.gvel = self.goal - self.pos  # the goal velocity of the agent
        self.gvel = self.gvel / (sqrt(self.gvel.dot(self.gvel))) * self.prefspeed
        self.maxspeed = max_speed
        self.radius = radius
        self.ksi = ksi
        self.dhor = dist_hor
        self.timehor = time_hor
        self.f = np.zeros(2)
        self.max_f = max_f

    def computeForces(self, neighbors: List[Any]) -> np.array:
        """ Compute forces for current agent based on the neighbors

        Args:
            neighbors (List[Agent]): List of all agents in the environment

        Returns:
            np.array: Vector describing the force to move
        """

        # Compute the goal force
        F_goal = (self.gvel - self.vel) / self.ksi
        self.f = F_goal

        # Calculate the neighbors that are within the sensing distance
        near_neighbors = [
            n for n in neighbors if n is not self and self.__distance_to(n) <= self.dhor
        ]

        # Iterate through all the neighbors
        for neighbor in near_neighbors:
            # Calculate the time to collision
            tau = self.time_to_collision(neighbor)

            # If the tau is infinite then the two agents will never collide
            if tau == float("inf"):
                continue

            # If tau is zero the agents are colliding now so go as fast as possible
            if tau == 0:
                F_a = self.max_f
            # Else the force is scaled by the maximum of the difference between the time
            # horizon and tau (> 0) over tau
            else:
                F_a = max(self.timehor - tau, 0) / tau

            # Calculate the vector that points exactly away from the collision
            n = (
                (self.pos + self.vel * tau) - (neighbor.pos + neighbor.vel * tau)
            ) / np.linalg.norm(self.pos + self.vel * tau)

            # Add the scaled vector to the force
            self.f += F_a * n

        return self.f

    def __distance_to(self, other: Any) -> float:
        """ Calculate the distance between two agents

        Args:
            other (Agent): Agent to calculate the distance to

        Returns:
            float: Distance
        """
        return np.linalg.norm(self.pos - other.pos)

    @staticmethod
    def __constrain(vector, limit):
        if vector[0] > 0:
            vector[0] = min(vector[0], limit)
        else:
            vector[0] = max(vector[0], -limit)

        if vector[1] > 0:
            vector[1] = min(vector[1], limit)
        else:
            vector[1] = max(vector[1], -limit)

        return vector

    def time_to_collision(self, other) -> float:
        """ Calculate time to collision if the two agents keep on the same path

        Args:
            other: Other agent to check against

        Returns:
            float: Time to collision
        """
        rad = self.radius + other.radius
        w = self.pos - other.pos
        c = w.dot(w) - pow(rad, 2)
        if c < 0:
            return 0
        v = self.vel - other.vel
        a = v.dot(v)
        b = w.dot(v)
        if b > 0:
            return float("inf")

        discr = pow(b, 2) - a * c
        if discr <= 0:
            return float("inf")

        tau = c / (-b + sqrt(discr))
        if tau < 0:
            return float("inf")

        return tau

    def update(self, dt):
        """
            Code to update the velocity and position of the agents.
            as well as determine the new goal velocity
        """
        if self.atGoal:
            return

        self.f = Agent.__constrain(self.f, self.max_f)
        self.vel += self.f * dt  # update the velocity

        # Cap the velocity
        self.vel = Agent.__constrain(self.vel, self.maxspeed)

        self.pos += self.vel * dt  # update the position

        # compute the goal velocity for the next time step. Do not modify this
        self.gvel = self.goal - self.pos
        dist_goal_sq = self.gvel.dot(self.gvel)
        if dist_goal_sq < self.goal_radius_sq:
            self.atGoal = True  # goal has been reached
        else:
            self.gvel = self.gvel / sqrt(dist_goal_sq) * self.prefspeed
