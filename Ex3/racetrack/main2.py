from generator import RacetrackGenerator
from environment import Environment
from montecarlo import MonteCarlo

gen = RacetrackGenerator()
racetrack = gen.racetrack1_from_book()
epsilon = 0.1
gamma = 1
env = Environment(gen, )
