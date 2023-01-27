from generator import RacetrackGenerator
from environment import Environment
from montecarlo import MonteCarlo
from agent import Agent
import matplotlib.pyplot as plt

gen = RacetrackGenerator()
racetrack = gen.racetrack1_from_book()
epsilon = 0.1
gamma = 1
velocity_constraint = 5
env = Environment(gen, 5)
mc = MonteCarlo(env,gen,velocity_constraint,gamma)
agent = Agent(epsilon,gen,env,mc)

episodes = 10 ** 5
rewards = []

for i in range(episodes+1):
    T = agent.learn_policy()
    S,A,B = agent.get_sab()
    t = mc.apply_mc_control(S, T, A, B)
    print("Episode {}: T={}, t={}, R={}".format(i, T, t, -1 * T))
    rewards.append(-1 * T)

k10_rewards = [rewards[i] for i in range(0, episodes + 1, 10000)]
plt.plot(k10_rewards)
plt.xlabel("Episodes (in ten thousands)")
plt.ylabel("Rewards")

# change based on which racetrack we are choosing
plt.title("Rewards from Racetrack 1")
plt.show()
