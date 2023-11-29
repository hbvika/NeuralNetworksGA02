# script to visualize how agent plays a game
# useful to study different iterations

import numpy as np
from agent import DeepQLearningAgent, PolicyGradientAgent, \
        AdvantageActorCriticAgent, HamiltonianCycleAgent, BreadthFirstSearchAgent, DeepQLearningAgent_torch
from game_environment import Snake, SnakeNumpy
from utils import visualize_game_torch
import json
# import keras.backend as K

# some global variables
version = 'v17.1'

with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames'] # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])

max_time_limit = 398

# setup the environment
env = Snake(board_size=board_size, frames=frames, max_time_limit=max_time_limit,
            obstacles=obstacles, version=version)
s = env.reset()
n_actions = env.get_num_actions()

agent = DeepQLearningAgent_torch(board_size=board_size, frames=frames, 
                           n_actions=n_actions, buffer_size=10, version=version)

models_to_load = ['torch_iter_1000', 'torch_iter_50000', 'torch_iter_100000', 'torch_iter_200000', 'torch_iter_400000']
for model in models_to_load:
    agent.load_model(file_path='models/{:s}'.format(model))
    
    for i in range(3):
        visualize_game_torch(env, agent,
            path='images/game_visual_{:s}_{:s}.mp4'.format(model,str(i)),
            debug=False, animate=True, fps=12)
