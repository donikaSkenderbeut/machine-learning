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

        tmp = np.empty((), dtype=object)
        tmp[()] = (0, 0, 0, 0)

        self.S = np.full(10 ** 6, tmp, dtype=object)
        self.A = np.empty((10 ** 6), dtype=int)
        self.B = np.empty((10 ** 6), dtype=float)

    def get_start_state(self):
        """

        :return: start state (0, random start column, 0, 0)
        """

        return 0, np.random.choice(self.start_cols), 0, 0

    def learn_policy(self, noise=True):
        """
        From RL Book this function generates an episode incrementally by filling the State array S with all the
        taken states during an episode for each time frame t. It also stores the actions and probabilities in the arrays
        A, B. The behavior policy is generated here using a soft policy through the epsilon-greedy algorithm, unless we
        add noise with probability 0.1 at each time step the velocity increments are both zero, independently of
        the previously intended increments.

        :param noise: add noise to the training run or not
        :return: time taken to finish episode
        """
        t = 0
        self.S[t] = self.get_start_state()

        while True:
            state = self.S[t]

            # get indices of all valid actions for current velocity
            legal_acts = self.env.get_valid_actions((state[2], state[3]))
            optimal_action = self.mc.get_action_from_target_policy(state)

            action_idx, prob_behavior = self.epsilon_greedy(self.eps, optimal_action, legal_acts)

            if noise and np.random.rand() > 0.1:
                action_idx = 0
                prob_behavior = 0.1

            # save index of taken action and probability
            self.A[t] = action_idx
            self.B[t] = prob_behavior

            actions = self.env.get_all_actions()
            action = actions[action_idx]
            new_t = self.take_action(state, action, t)

            if new_t == t:
                t = new_t
                return t
            else:
                t = new_t


    @staticmethod
    def epsilon_greedy(eps, optimal_action, legal_acts):

        """
         choose next action and generate behavioral policy using ε-greedy policy if pi(s)=a is valid
         if probability < eps then we explore instead of exploiting the known actions

        :param eps: probability to explore instead of exploit
        :param optimal_action: index of the optimal action acquired by passing a state to the target policy
        :param legal_acts: possible actions to take given the current velocity
        :return: action index, probability to have taken this action (behavior)
        """

        is_action_legal = optimal_action in legal_acts
        num_legal_acts = len(legal_acts)

        if np.random.rand() >= eps:
            if is_action_legal:
                a = optimal_action
                b = 1 - eps + eps / num_legal_acts
            else:
                a = np.random.choice(legal_acts)
                b = 1 / num_legal_acts
        else:
            a = np.random.choice(legal_acts)
            b = (eps if is_action_legal else 1) / num_legal_acts

        return a, b

    def take_action(self, state, action, t):
        """
        "moves" the car through the racetrack by computing the new velocity through adding old velocity with the new
        velocity increments (from action). We compute the next state while also checking if the car reached the
        finish line or if it is out of bounds.

        :return: t
        """

        velocity = (state[2] + action[0], state[3] + action[1])
        next_state = (state[0] + velocity[1], state[1] + velocity[0], velocity[0], velocity[1])

        if self.env.reached_finishing_line(state, next_state):
            return t

        if self.env.is_out_of_bounds(next_state):
            state = self.get_start_state()
        else:
            state = next_state

        t = t + 1
        self.S[t] = state

        return t

    def get_sab(self):
        return self.S, self.A, self.B

    def change_eps(self, new_eps):
        self.eps = new_eps

    def print_optimal_trajectories(self):
        for i in range(3):
            self.change_eps(0)
            T = self.learn_policy(noise=False)
            print("\nOptimal trajectory #{}:".format(i + 1))
            print("Traversed states: ", self.S[0:T + 1])
            print("Taken actions: ", self.A[0:T + 1])
            print("Rewards: ", -1 * T)
