import numpy as np
from generator import RacetrackGenerator
from environment import Environment
from montecarlo import MonteCarlo


class Agent:

    def __init__(self, epsilon, racetrack: RacetrackGenerator, env: Environment, mc: MonteCarlo):
        self.start_cols, self.finish_coords = racetrack.get_start_finish_line()
        self.env = env
        self.mc = mc
        self.eps = epsilon

    def get_start_state(self):
        return 0, np.random.choice(self.start_cols), 0, 0

    def learn_policy(self):
        t = 0
        S = []
        A = []
        B = []
        S[t] = self.get_start_state()

        while True:
            s = S[t]

            # get indices of all valid actions for current velocity
            legal_acts = self.env.get_valid_actions((s[2], s[3]))
            num_legal_acts = len(legal_acts)
            optimal_action = self.mc.get_action_from_target_policy(s)
            is_a_legal = optimal_action in legal_acts


    @staticmethod
    def epsilon_greedy(eps, is_a_legal, num_legal_acts, optimal_action, legal_acts):
        if np.random.rand() >= eps:
            if is_a_legal:
                a = optimal_action
                b = 1 - eps + eps / num_legal_acts
            else:
                a = np.random.choice(legal_acts)
                b = 1 / num_legal_acts
        else:
            a = np.random.choice(legal_acts)
            b = (eps if is_a_legal else 1) / num_legal_acts


