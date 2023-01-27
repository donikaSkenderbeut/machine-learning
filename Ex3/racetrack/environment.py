import random
import math
import numpy as np
from generator import RacetrackGenerator


class Environment:

    def __init__(self, generator: RacetrackGenerator, racetrack: np.ndarray, velocity_constraint: int):
        self.generator = generator
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
        finish_line = self.generator.get_start_finish_line()[1]
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

    def is_out_of_bounds(self, next_s):
        rows = self.generator.get_dimensions()[0]
        cols = self.generator.get_dimensions()[1]

        # check if out of bounds or hit X
        if not (0 <= next_s[0] < rows) or not (0 <= next_s[1] < cols) \
                or self.racetrack[rows - 1 - next_s[0]][next_s[1]] == 'X':
            return True

        return False

    def get_all_actions(self):
        return self.actions
