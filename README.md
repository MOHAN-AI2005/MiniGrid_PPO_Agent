# MiniGrid PPO Agent

This project contains a simple implementation of a PPO (Proximal Policy Optimization) agent trained in the MiniGrid environment using gym-minigrid.

## Features
- Lightweight PPO agent
- Runs on MiniGrid's `Empty-5x5-v0` environment
- Logs training reward with TensorBoard

## Requirements
- gym
- gym-minigrid
- torch
- tensorboard

## Running
```bash
pip install -r requirements.txt
python src/train_agent.py
tensorboard --logdir runs/
```
