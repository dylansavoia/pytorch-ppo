# import mujoco_py
import gymnasium as gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from ppo_agent import PPOAgent
import sys, os, json
from glob import glob
import argparse
from pprint import pprint

# Environemnt Params
MODELS_PATH        = 'models'
DEFAULT_EPISODES   = 2000
DEFAULT_MAX_STEPS  = 2000
DEFAULT_CHECKPOINT = 5

## These may be updated by arguments
EnvName = 'MountainCarContinuous-v0'
TRAIN = False
RENDER = True

writer = SummaryWriter(max_queue=2)

def main():
    global TRAIN, RENDER, EnvName

    # Training Parameters
    params_path = os.path.join(MODELS_PATH, f'{EnvName}.json')
    with open(params_path, 'r') as f:
        params = json.load(f)

    EPOCHS        = params["EPOCHS"] if "EPOCHS" in params else 200
    LR            = params["LR"] if "LR" in params else 1e-3
    C2            = params["C2"] if "C2" in params else 0
    GAMMA         = params["GAMMA"] if "GAMMA" in params else 0.99
    STD           = params["STD"] if "STD" in params else 1
    NETSIZE       = params["NETSIZE"] if "NETSIZE" in params else 64
    BATCHSIZE     = params["BATCHSIZE"] if "BATCHSIZE" in params else 500
    TRAJECTORIES  = params["TRAJECTORIES"] if "TRAJECTORIES" in params else 10
    MAX_STEPS     = params["MAX_STEPS"] if "MAX_STEPS" in params else 2000
    CHECKPOINT    = params["CHECKPOINT"] if "CHECKPOINT" in params else 5
    EPISODES      = params["EPISODES"] if "EPISODES" in params else 2000

    print("Environment:  ", EnvName)
    print("Train:        ", TRAIN)
    print("EPOCHS:       ", EPOCHS)
    print("LR:           ", LR)
    print("C2:           ", C2)
    print("GAMMA:        ", GAMMA)
    print("STD:          ", STD)
    print("NETSIZE:      ", NETSIZE)
    print("BATCHSIZE:    ", BATCHSIZE)
    print("TRAJECTORIES: ", TRAJECTORIES)
    print("MAX_STEPS:    ", MAX_STEPS)
    print("CHECKPOINT:   ", CHECKPOINT)
    print("EPISODES:     ", EPISODES)

    # save model after collecting N trajectories 
    # (which corresponds to when the update is calculated)
    SAVE_STEP = CHECKPOINT * TRAJECTORIES
    save_model_name = os.path.join(MODELS_PATH, EnvName + ".pth")

    total = 0

    env = gym.make(EnvName, render_mode="human" if RENDER else None)
    agent = PPOAgent(
            TRAIN, env=env,
            lr=LR, c2=C2,
            net_size=NETSIZE,
            net_std=STD,
            y=GAMMA,
            traj=TRAJECTORIES,
            bs=BATCHSIZE,
            ep=EPOCHS,
            W=writer
    )
    if not TRAIN: agent.load(save_model_name)


    for i in range(EPISODES):
        state, _ = env.reset()

        for t in range(MAX_STEPS+1):
            # RL Step
            action = agent(state)
            new_state, reward, done, _, _ = env.step(action)
            
            # Impose done=True if last-step
            if t == MAX_STEPS: done = True

            agent.observe(state, reward, new_state, done, i)

            total += reward
            state  = new_state

            if done: break

        # Print Performance
        print(f"[{i}] Steps: {t}\tReward: {total}")
        writer.add_scalar('Reward', total, i)
        total = 0

        if TRAIN and (i % SAVE_STEP) == SAVE_STEP -1:
            agent.save(save_model_name)
            print("Model Checkpoint saved")

    env.close()


envs_names = glob(f'{MODELS_PATH}/*.json')
envs_names = [x.split('/')[-1].split('.')[0] for x in envs_names]

parser = argparse.ArgumentParser(description="Train PPO models and run Gym environmnts")
parser.add_argument('env', type=str, metavar="environment", help=", ".join(envs_names),
                    choices=envs_names, default="MountainCarContinuous-v0")
parser.add_argument('--train', action='store_true')
args = parser.parse_args()

EnvName = args.env
TRAIN   = args.train
RENDER  = not(TRAIN)

main()
