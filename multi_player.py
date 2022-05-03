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


def main(_):
    game = "quoridor"
    print("Choose number of players...(2-4)")
    num_players = int(input())
    print("Choes size of board...(3-100)")
    board_size = int(input())
    print("Choose number of walls...(0-100)")
    wall_count = int(input())
    params = {"wall_count": wall_count, "board_size": board_size, "num_players": num_players}
    policy = "ql"

    env = rl_environment.Environment(
        game,
        board_size=params["board_size"],
        wall_count=params["wall_count"],
        players=params["num_players"],
        ansi_color_output=True,
    )
    #print(f"size state space : {len(list(trained_agents[0]._q_values.keys()))}")

    while True:
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            print(env.get_state)
            action = command_line_action(time_step, env.get_state)
            time_step = env.step([action])

        print(env.get_state)

        logging.info("End of game!")
        for i in range(params['num_players']):
            if time_step.rewards[i] > 0:
                logging.info(f"Player {i} won")


if __name__ == "__main__":
    app.run(main)
