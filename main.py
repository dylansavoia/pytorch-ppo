# import mujoco_py
import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from ppo_agent import PPOAgent
import sys, os, json
from glob import glob

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

    Tot = 0

    env = gym.make(EnvName)
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
        s = env.reset()

        for t in range(MAX_STEPS):
            if RENDER: env.render()

            # RL Step
            a = agent(s)
            s1, r, done, _ = env.step(a)
            agent.observe(s, r, s1, done, i+1)

            Tot += r
            s = s1

            if done: break

        # Print Performance
        print("[%d] Steps: %d\tReward: %d" % (i, t+1, Tot))
        writer.add_scalar('Reward', Tot, i)
        Tot = 0

        if TRAIN and (i % SAVE_STEP) == SAVE_STEP -1:
            agent.save(save_model_name)
            print("Model Checkpoint saved")

    env.close()

def checkArguments():
    global EnvName, TRAIN, RENDER
    print_usage = False
    N = len(sys.argv)

    if N == 1: return True

    envs_names = glob(f'{MODELS_PATH}/*.json')
    envs_names = [x.split('/')[-1].split('.')[0] for x in envs_names]

    if N > 1: env = sys.argv[1]
    if N > 2: TRAIN = sys.argv[2].lower() == 'true'
    RENDER = not TRAIN

    if env in envs_names:
        EnvName = env
    else:
        print("Usage: python main.py <env_name> <train>")
        print("<env_name>:\n\t" + "\n\t".join(envs_names))
        print("")
        print("<train>: True or False")


checkArguments()
print(f'EnvName: {EnvName}')
print(f'Train: {TRAIN}')
main()
