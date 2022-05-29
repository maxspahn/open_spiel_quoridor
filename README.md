# Learning Quoridor

In this repository, open_spiel is used to train an agent to play Quoridor.

My agent is still very bad (Qlearning), how would you approach it?

## Install tensorflow with gpu support
https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d

## Dependencies

You must have python installed as well as cmake and a cpp compiler.

```bash
sudo apt install cmake clang
```

## Installation

Best is to install via [poetry](https://python-poetry.org/docs/) 
as it automatically sets up a virtual environment.

```bash
poetry install
```

## Multiplayer gaming through

```bash
poetry shell
python3 multi_player.py
```

## Policies

Trained policies are not uploaded.
