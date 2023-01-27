import numpy as np
from generator import RacetrackGenerator
from environment import Environment
import matplotlib.pyplot as plt

rows, cols = 32, 18
#rows, cols = 30, 33

gen = RacetrackGenerator()
racetrack = gen.racetrack1_from_book()
start_cols = range(4, 10)
#start_cols = range(1, 24)
fin_cells = {(26, cols - 1), (27, cols - 1), (28, cols - 1), (29, cols - 1), (30, cols - 1), (31, cols - 1)}
#fin_cells = {(21, cols-1), (22, cols-1), (23, cols-1), (24, cols-1), (25, cols-1), (26, cols-1), (27, cols-1),
#             (28, cols-1), (29, cols-1)}

epsilon = 0.1
gamma = 1

# actions: velocity increments at a given time, 9 actions 0, 1, -1
actions = [
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
act_len = len(actions)

# 5 velocity values 0 to 4
vel_len = 5

# compute all valid actions for each velocity combination i.e vertical 0 horizontal 0, v 1 h 0 ...
# Both velocity components are restricted to be non-negative and less than 5, and they cannot both be zero.
valid_acts = [
    [actions.index(a) for a in actions if (h + a[0]) in range(5) and (v + a[1]) in range(5) and not ((h + a[0]) == 0 and
                                                                                                     (v + a[1]) == 0)]
    for h in range(vel_len)
    for v in range(vel_len)
]

# put values in dict for ease of accessibility
dict_valid_acts = {}
j = 0
for h in range(vel_len):
    for v in range(vel_len):
        dict_valid_acts[(h, v)] = valid_acts[j]
        j += 1

print(dict_valid_acts)
env = Environment(racetrack)

# initialise Q, C and pi
# Q is the value matrix given state and action
# C is a cumulative sum of the weights
# pi is the target policy
# state is a tuple made of 4 components (x-coordinate, y-coordinate, vertical velocity, horizontal velocity)
Q = np.random.rand(rows, cols, vel_len, vel_len, act_len) * -400  # broader range of numbers makes computation faster
C = np.zeros((rows, cols, vel_len, vel_len, act_len))
pi = np.ones((rows, cols, vel_len, vel_len), dtype=int)
for r in range(rows):
    for c in range(cols):
        for h in range(vel_len):
            for v in range(vel_len):
                pi[r, c, h, v] = np.argmax(Q[r, c, h, v])

# initialize state, action, behaviour policy probability arrays
tmp = np.empty((), dtype=object)
tmp[()] = (0, 0, 0, 0)
S = np.full(10 ** 6, tmp, dtype=object)
A = np.empty((10 ** 6), dtype=int)
B = np.empty((10 ** 6), dtype=float)


def learn_policy(eps, noise=True):
    t = 0

    # start state (0, random start column, 0 v-velocity, 0 h-velocity)
    S[t] = (0, np.random.choice(start_cols), 0, 0)

    while True:
        s = S[t]
        # get indices of all valid actions for current velocity
        acts = dict_valid_acts[(s[2], s[3])]
        num_acts = len(acts)

        # choose next action and generate behavioral policy using ε-greedy policy if pi(s)=a is valid
        # otherwise choose randomly, and save its probability
        pi_a = pi[s[0], s[1], s[2], s[3]]
        pi_a_valid = pi_a in acts  # if pi(s) returns valid index that can be found in the valid action then True
        if np.random.rand() >= eps:
            if pi_a_valid:
                a = pi_a
                b = 1 - eps + eps / num_acts
            else:
                a = np.random.choice(acts)
                b = 1 / num_acts
        else:
            a = np.random.choice(acts)
            b = (eps if pi_a_valid else 1) / num_acts

        # add some noise
        # with probability 0.1 at each time step the velocity increments are
        # both zero, independently of the intended increments
        if noise and np.random.rand() < 0.1:
            a = 0
            b = 0.1

        # save index of taken action and probability
        A[t] = a
        B[t] = b
        act = actions[a]

        # next state
        new_vel = (s[2] + act[0], s[3] + act[1])
        next_s = (s[0] + new_vel[1], s[1] + new_vel[0], new_vel[0], new_vel[1])

        # check if car hits finish line by computing "path", which is something between s and next_s, thus checking if
        # any of the driven paths from s to next_s crossed the finish line
        path = set([
            (min(s[0] + i, next_s[0]), min(s[1] + i, next_s[1]))
            for i in range(max(new_vel[0], new_vel[1]))
        ])
        if len(path.intersection(fin_cells)) > 0:
            return t

        # check if car hits boundary
        if not (0 <= next_s[0] < rows) or not (0 <= next_s[1] < cols) or racetrack[rows - 1 - next_s[0]][next_s[1]] \
                == 'X':
            # go back to start
            s = (0, np.random.choice(start_cols), 0, 0)
        else:
            s = next_s

        t += 1
        S[t] = s


def apply_mc_control(T):
    G = 0.0
    W = 1.0
    R = -1

    for t in range(T - 1, 0, -1):
        s = S[t]

        G = gamma * G + R
        C[s[0], s[1], s[2], s[3], A[t]] += W
        Q[s[0], s[1], s[2], s[3], A[t]] += W * (G - Q[s[0], s[1], s[2], s[3], A[t]]) / C[s[0], s[1], s[2], s[3], A[t]]

        acts = dict_valid_acts[(s[2], s[3])]

        # from Q choose the index of the action in given state which returns the highest value
        # and update the target policy accordingly
        pi[s[0], s[1], s[2], s[3]] = acts[np.argmax(Q[s[0], s[1], s[2], s[3], :][acts])]

        # if chosen action at given timestamp does not comply with the target policy then proceed to next episode
        if A[t] != pi[s[0], s[1], s[2], s[3]]:
            return t

        W /= B[t]
    return 0


# print several trajectories following the optimal policy
def optimal_trajectories():
    for i in range(3):
        T = learn_policy(0.0, noise=False)
        print("\nOptimal trajectory #{}:".format(i + 1))
        print("S: ", S[0:T + 1])
        print("A: ", A[0:T + 1])
        print("R: ", -1 * T)


episode_num = 10 ** 5
rewards = []

for i in range(0, episode_num + 1):
    T = learn_policy(epsilon)
    t = apply_mc_control(T)
    print("Episode {}: T={}, t={}, R={}".format(i, T, t, -1 * T))
    rewards.append(-1 * T)

k10_rewards = [rewards[i] for i in range(0, episode_num + 1, 10000)]
plt.plot(k10_rewards)
plt.xlabel("Episodes (in ten thousands)")
plt.ylabel("Rewards")

# change based on which racetrack we are choosing
plt.title("Rewards from Racetrack 1")
optimal_trajectories()
plt.show()
