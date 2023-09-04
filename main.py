# import mujoco_py
import gymnasium as gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from ppo_agent import PPOAgent
import sys, os, json
from glob import glob
import argparse

# Environemnt Params
MODELS_PATH = 'models'
EPISODES = 2000
MAX_STEPS = 2000

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

    EPOCHS  = params["EPOCHS"]
    LR      = params["LR"]
    C2      = params["C2"]
    GAMMA   = params["GAMMA"]
    STD     = params["STD"]
    NETSIZE = params["NETSIZE"]
    BATCHSIZE    = params["BATCHSIZE"]
    TRAJECTORIES = params["TRAJECTORIES"]

    # save model after collecting N trajectories 
    # (which corresponds to when the update is calculated)
    SAVE_STEP = 5 * TRAJECTORIES
    save_model_name = os.path.join(MODELS_PATH, EnvName + ".model")

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
            action = agent(state.T)[0]
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

print(f'EnvName: {EnvName}')
print(f'Train: {TRAIN}')
main()
