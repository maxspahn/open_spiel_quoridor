# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MCTS example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import sys
from typing import Dict, Any

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.algorithms import mcts
from open_spiel.python.bots import gtp
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel

import optuna

_KNOWN_PLAYERS = [
    # A generic Monte Carlo Tree Search agent.
    "mcts",

    # A generic random agent.
    "random",

    # You'll be asked to provide the moves.
    "human",

    # Run an external program that speaks the Go Text Protocol.
    # Requires the gtp_path flag.
    "gtp",
]

flags.DEFINE_string("game", "quoridor", "Name of the game.")
flags.DEFINE_enum("player1", "mcts", _KNOWN_PLAYERS, "Who controls player 1.")
flags.DEFINE_enum("player2", "mcts", _KNOWN_PLAYERS, "Who controls player 2.")
flags.DEFINE_enum("player3", "mcts", _KNOWN_PLAYERS, "Who controls player 3.")
flags.DEFINE_enum("player4", "mcts", _KNOWN_PLAYERS, "Who controls player 4.")
flags.DEFINE_string("gtp_path", None, "Where to find a binary for gtp.")
flags.DEFINE_multi_string("gtp_cmd", [], "GTP commands to run at init.")
flags.DEFINE_string("az_path", None,
                    "Path to an alpha_zero checkpoint. Needed by an az player.")
flags.DEFINE_integer("uct_c", 2, "UCT's exploration constant.")
flags.DEFINE_integer("rollout_count", 1, "How many rollouts to do.")
flags.DEFINE_integer("max_simulations", 10000, "How many simulations to run.")
flags.DEFINE_integer("num_games", 100, "How many games to play.")
flags.DEFINE_integer("seed", None, "Seed for the random number generator.")
flags.DEFINE_bool("random_first", False, "Play the first move randomly.")
flags.DEFINE_bool("solve", True, "Whether to use MCTS-Solver.")
flags.DEFINE_bool("quiet", False, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")

FLAGS = flags.FLAGS


def _opt_print(*args, **kwargs):
  pass

def _init_mcts_bot(game, params):
  uct_c = params['uct_c']
  max_simulations = params['max_simulations']
  rollout_count = params['rollout_count']
  rng = np.random.RandomState()
  evaluator = mcts.RandomRolloutEvaluator(rollout_count, rng)
  return mcts.MCTSBot(
    game,
    uct_c,
    max_simulations,
    evaluator,
    random_state=rng,
    solve=True,
    verbose=False)

def _init_random_bot(player_id):
  rng = np.random.RandomState()
  return uniform_random.UniformRandomBot(player_id, rng)

def _init_human_bot():
    return human.HumanBot()

def _get_action(state, action_str):
  for action in state.legal_actions():
    if action_str == state.action_to_string(state.current_player(), action):
      return action
  return None


def _play_game(game, bots):
  """Plays one game."""
  state = game.new_initial_state()
  _opt_print("Initial state:\n{}".format(state))

  history = []

  while not state.is_terminal():
    current_player = state.current_player()
    # The state can be three different types: chance node,
    # simultaneous node, or decision node
    if state.is_chance_node():
      # Chance node: sample an outcome
      outcomes = state.chance_outcomes()
      num_actions = len(outcomes)
      _opt_print("Chance node, got " + str(num_actions) + " outcomes")
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(action_list, p=prob_list)
      action_str = state.action_to_string(current_player, action)
      _opt_print("Sampled action: ", action_str)
    elif state.is_simultaneous_node():
      raise ValueError("Game cannot have simultaneous nodes.")
    else:
      # Decision node: sample action for the single current player
      bot = bots[current_player]
      action = bot.step(state)
      action_str = state.action_to_string(current_player, action)
      _opt_print("Player {} sampled action: {}".format(current_player,
                                                       action_str))

    for i, bot in enumerate(bots):
      if i != current_player:
        bot.inform_action(state, current_player, action)
    history.append(action_str)
    state.apply_action(action)

    _opt_print("Next state:\n{}".format(state))

  # Game is now done. Print return for each player
  returns = state.returns()
  _opt_print("Returns:", " ".join(map(str, returns)), ", Game actions:",
        " ".join(history))

  for bot in bots:
    bot.restart()

  return returns, history

def sample_mcts_params(trial: optuna.Trial) -> Dict[str, Any]:
  uct_c = trial.suggest_float("uct_c", 1, 5, log = False)
  rollout_count = trial.suggest_int("rollout_count", 1, 5)
  max_simulations = trial.suggest_int("max_simulations", 100, 300, log = False)
  return {'max_simulations': max_simulations, 'uct_c' : uct_c, 'rollout_count': rollout_count}


def objective(trial: optuna.Trial) -> float:
  num_players = 4
  board_size = 5
  wall_count = 2
  params_game = {"wall_count": wall_count, "board_size": board_size, "players": num_players}
  game = pyspiel.load_game('quoridor', params_game)
  params = sample_mcts_params(trial)
  mcts_agent = _init_mcts_bot(game, params)
  params_bot = {'max_simulations': 20, 'uct_c': 3, 'rollout_count': 2}
  mcts_bot = _init_mcts_bot(game, params_bot)
  bots = [mcts_agent, mcts_bot, _init_random_bot(2), _init_random_bot(3)]
  nb_games = 3
  overall_returns = []
  for i in range(nb_games):
      print(f"playing game {i}")
      returns, history = _play_game(game, bots)
      overall_returns.append(returns[0])
  print(overall_returns)
  return np.mean(overall_returns)



def main(argv):
  #print("Choose number of players...(2-4)")
  #num_players = int(input())
  num_players = 4
  #print("Choose size of board...(3-100)")
  #board_size = int(input())
  board_size = 3
  #print("Choose number of walls...(0-100)")
  #wall_count = int(input())
  wall_count = 0
  params = {"wall_count": wall_count, "board_size": board_size, "players": num_players}
  game = pyspiel.load_game(FLAGS.game, params)
  if game.num_players() > 4:
    sys.exit("This game requires more players than the example can handle.")
  bots = [
      _init_bot(FLAGS.player1, game, 0),
      _init_bot(FLAGS.player2, game, 1),
      _init_bot(FLAGS.player3, game, 2),
      _init_bot(FLAGS.player4, game, 3),
  ]
  bots = bots[:num_players]
  histories = collections.defaultdict(int)
  overall_returns = [0, 0, 0, 0]
  overall_wins = [0, 0, 0, 0]
  game_num = 0
  try:
    for game_num in range(FLAGS.num_games):
      returns, history = _play_game(game, bots, argv[1:])
      histories[" ".join(history)] += 1
      for i, v in enumerate(returns):
        overall_returns[i] += v
        if v > 0:
          overall_wins[i] += 1
      print(overall_wins)
  except (KeyboardInterrupt, EOFError):
    game_num -= 1
    print("Caught a KeyboardInterrupt, stopping early.")
  print("Number of games played:", game_num + 1)
  print("Number of distinct games played:", len(histories))
  print("Players:", FLAGS.player1, FLAGS.player2, FLAGS.player3, FLAGS.player4)
  print("Overall wins", overall_wins)
  print("Overall returns", overall_returns)


if __name__ == "__main__":
    # Let us minimize the objective function above.
    print("Running 10 trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    # We can continue the optimization as follows.
    print("Running 20 additional trials...")
    study.optimize(objective, n_trials=20)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    """
    # We can specify the timeout instead of a number of trials.
    print("Running additional trials in 2 seconds...")
    study.optimize(objective, timeout=2.0)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
    """
