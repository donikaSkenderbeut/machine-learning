from environment import Environment
from generator import RacetrackGenerator
import numpy as np

class MonteCarlo:
    def __init__(self, env:Environment, generator:RacetrackGenerator, velocity_constraint: int, gamma:int):
        # initialise Q, C and pi
        # Q is the value matrix given state and action
        # C is a cumulative sum of the weights
        # pi is the target policy
        # state is a tuple made of 4 components (x-coordinate, y-coordinate, vertical velocity, horizontal velocity)
        vel_len = velocity_constraint
        act_len = len(env.get_all_actions())
        rows = generator.get_dimensions()[0]
        cols = generator.get_dimensions()[1]
        self.env = env
        self.gamma = gamma
        self.Q = np.random.rand(rows, cols, vel_len, vel_len, act_len) * -400  # broader range of numbers makes computation faster
        self.C = np.zeros((rows, cols, vel_len, vel_len, act_len))
        self.pi = np.ones((rows, cols, vel_len, vel_len), dtype=int)
        for r in range(rows):
            for c in range(cols):
                for h in range(vel_len):
                    for v in range(vel_len):
                        self.pi[r, c, h, v] = np.argmax(self.Q[r, c, h, v])

    def apply_mc_control(self, S, T, A, B):
        G = 0.0
        W = 1.0
        R = -1

        for t in range(T - 1, 0, -1):
            s = S[t]

            G = self.gamma * G + R
            self.C[s[0], s[1], s[2], s[3], A[t]] += W
            self.Q[s[0], s[1], s[2], s[3], A[t]] += W * (G - self.Q[s[0], s[1], s[2], s[3], A[t]]) / self.C[s[0], s[1], s[2], s[3], A[t]]

            acts = self.env.get_valid_actions(s)

            # from Q choose the index of the action in given state which returns the highest value
            # and update the target policy accordingly
            self.pi[s[0], s[1], s[2], s[3]] = acts[np.argmax(self.Q[s[0], s[1], s[2], s[3], :][acts])]

            # if chosen action at given timestamp does not comply with the target policy then proceed to next episode
            if A[t] != self.pi[s[0], s[1], s[2], s[3]]:
                return t

            W /= B[t]
        return 0


    def get_action_from_target_policy(self, state):
        return self.pi[state[0], state[1], state[2], state[3]]
