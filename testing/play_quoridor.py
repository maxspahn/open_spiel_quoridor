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
from absl import app
from absl import flags
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(1e4), "Number of train episodes.")
flags.DEFINE_boolean(
    "iteractive_play",
    True,
    "Whether to run an interactive play with the agent after training.",
)


def command_line_action(time_step, state):
    """Gets a valid action from the user on the command line."""
    current_player = time_step.observations["current_player"]
    legal_actions = time_step.observations["legal_actions"][current_player]
    action = -1
    while action not in legal_actions:
        action_map = {
            state.action_to_string(action): action for action in legal_actions
        }
        print("Choose an action from {}:".format(action_map))
        sys.stdout.flush()
        action_str = input()
        try:
            # action = int(action_str)
            action = action_map[str(action_str)]
        except KeyError:
            print("Invalid action")
            continue
    return action


def load_agent(params, policy):
    import pickle
    fileName = '../policies/' + policy
    for item in params:
        fileName += "_" + item + str(params[item])
    fileName += ".pickle"
    with open(fileName, "rb") as file:
        agent = pickle.load(file)
    return agent


def main(_):
    game = "quoridor"
    params = {"wall_count": 2, "board_size": 5}
    policy = "ql"

    env = rl_environment.Environment(
        game, board_size=params["board_size"], wall_count=params["wall_count"]
    )
    trained_agents = load_agent(params, policy)
    #print(f"size state space : {len(list(trained_agents[0]._q_values.keys()))}")

    # 2. Play from the command line against the trained agent.
    human_player = 1
    while True:
        logging.info("You are playing as %s", "O" if human_player else "X")
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            if player_id == human_player:
                # agent_out = agents[human_player].step(time_step, is_evaluation=True)
                # logging.info("\n%s", agent_out.probs.reshape((3, 3)))
                # logging.info("\n%s", pretty_board(time_step))
                print(env.get_state)
                action = command_line_action(time_step, env.get_state)
            else:
                agent_out = trained_agents[player_id].step(
                    time_step, is_evaluation=True
                )
                action = agent_out.action
            time_step = env.step([action])

        print(env.get_state)

        logging.info("End of game!")
        if time_step.rewards[human_player] > 0:
            logging.info("You win")
        elif time_step.rewards[human_player] < 0:
            logging.info("You lose")
        else:
            logging.info("Draw")
        # Switch order of players
        human_player = 1 - human_player


if __name__ == "__main__":
    app.run(main)
