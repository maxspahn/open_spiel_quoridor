# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tabular Q-Learner example on Tic Tac Toe.

Two Q-Learning agents are trained by playing against each other. Then, the game
can be played against the agents from the command line.

After about 10**5 training episodes, the agents reach a good policy: win rate
against random opponents is around 99% for player 0 and 92% for player 1.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import time
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment, rl_tools
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.pytorch import dqn

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(1e8), "Number of train episodes.")
flags.DEFINE_boolean(
    "iteractive_play",
    True,
    "Whether to run an interactive play with the agent after training.",
)


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    wins = np.zeros(2)
    for player_pos in range(2):
        if player_pos == 0:
            cur_agents = [trained_agents[0], random_agents[1]]
        else:
            cur_agents = [random_agents[0], trained_agents[1]]
        for i in range(num_episodes):
            time_step = env.reset()
            while not time_step.last():
                if i == -1:
                    print(f"trained agent at position {player_pos}")
                    time.sleep(1)
                    print(env.get_state)
                player_id = time_step.observations["current_player"]
                agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                time_step = env.step([agent_output.action])
            if time_step.rewards[player_pos] > 0:
                wins[player_pos] += 1
    return wins / num_episodes


def save_agent(agent, params, policy):
    import pickle

    fileName = "../policies/" + policy
    for item in params:
        fileName += "_" + item + str(params[item])
    fileName += ".pickle"
    with open(fileName, "wb") as file:
        pickle.dump(agent, file)


def load_agent(params, policy):
    import pickle

    fileName = "../policies/" + policy
    for item in params:
        fileName += "_" + item + str(params[item])
    fileName += ".pickle"
    try:
        with open(fileName, "rb") as file:
            agent = pickle.load(file)
    except FileNotFoundError:
        return None
    return agent


def main(_):
    game = "quoridor"
    num_players = 2
    params = {"wall_count": 2, "board_size": 5}
    policy = "ql"

    env = rl_environment.Environment(
        game, board_size=params["board_size"], wall_count=params["wall_count"]
    )
    num_actions = env.action_spec()["num_actions"]
    state_size = env.observation_spec()["info_state"][0]

    agents = load_agent(params, policy)

    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    if not agents:
        if policy == "ql":
            agents = [
                tabular_qlearner.QLearner(
                    player_id=0, num_actions=num_actions, discount_factor=0.1, epsilon_schedule=rl_tools.ConstantSchedule(0.5)
                ),
                tabular_qlearner.QLearner(
                    player_id=1, num_actions=num_actions, discount_factor=0.1, epsilon_schedule=rl_tools.ConstantSchedule(0.5)
                ),
            ]
        elif policy == "dql":
            q_agents = load_agent(params, "ql")
            agents = [
                dqn.DQN(  # pylint: disable=g-complex-comprehension
                    0,
                    state_representation_size=state_size,
                    num_actions=num_actions,
                    hidden_layers_sizes=[4048, 1024],
                    replay_buffer_capacity=50,
                    batch_size=20,
                ),
                dqn.DQN(  # pylint: disable=g-complex-comprehension
                    1,
                    state_representation_size=state_size,
                    num_actions=num_actions,
                    hidden_layers_sizes=[4048, 1024],
                    replay_buffer_capacity=50,
                    batch_size=20,
                    learning_rate=0.01,
                ),
            ]
    else:
        print("using pre-trained agents")

    # 1. Train the agents
    training_episodes = FLAGS.num_episodes
    for cur_episode in range(training_episodes):
        if cur_episode % int(1e3) == 0:
            win_rates = eval_against_random_bots(env, agents, random_agents, 100)
            logging.info("Starting episode %s, win_rates %s", cur_episode, win_rates)
            save_agent(agents, params, policy)
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            time_step = env.step([agent_output.action])

        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)


if __name__ == "__main__":
    app.run(main)
