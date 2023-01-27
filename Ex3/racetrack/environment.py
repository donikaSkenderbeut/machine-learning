import random
import math
import numpy as np
##from agent import Agent
from generator import RacetrackGenerator


class Environment:

    def __init__(self, racetrack:RacetrackGenerator, velocity_constraint:int):
        self.racetrack = racetrack
        self.actions = [
            (0, 0),
            (0, 1),
            (0, -1),
            (1, 0),
            (1, 1),
            (1, -1),
            (-1, 0),
            (-1, 1),
            (-1, -1),
        ]
        # 5 velocity values 0 to 4
        self.velocity_constraint = velocity_constraint

        # put valid action for each possible velocity state (horizontally, vertically)
        self.dict_valid_acts = {}
        for h in range(self.velocity_constraint):
            for v in range(self.velocity_constraint):
                v_actions = []
                for a in self.actions:
                    if ((h + a[0]) in range(5) and (v + a[1]) in range(5)) and not (
                            (h + a[0]) == 0 and (v + a[1]) == 0):
                        v_actions.append(self.actions.index(a))

                self.dict_valid_acts[(h, v)] = v_actions


    def get_valid_actions(self, agend_velocity):
        return self.dict_valid_acts[agend_velocity]

    def reached_finishing_line(self, state, next_state):
        finish_line = self.racetrack.get_start_finish_line()[1]
        points = []

        x = state[0]
        for y in range(state[1], next_state[1] + 1):

            points.append((x, y))
            if x != next_state[0]:
                x += 1

        if any(i in points for i in finish_line):
            return True
        else:
            return False

    def is_out_of_bounds(self, state, next_state):
        return False

    def get_all_actions(self):
        return self.actions;

