# Proximal Policy Optimization
A py-torch implementation of the PPO algorithm.

The code is made of two files: ppo_agent.py which implements the PPO algorithm
itself, and main.py which renders the chosen environment and runs the
agent on it.

**Environemnts must support continuous action spaces.**


## Install
Using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):
```bash
$ conda env create --file environment.yml
```

## Train and Run Environments
It is possible to run the code with the following command:
```bash
$ python main.py -h
usage: main.py [-h] [--train] environment

Train PPO models and run Gym environmnts

positional arguments:
  environment  BipedalWalker-v3, MountainCarContinuous-v0, Pendulum-v1

options:
  -h, --help   show this help message and exit
  --train
```

The train flag specifies whether training has to occur or not, otherwise the
most recent saved parameters will be used to show performance on the learned
environment.

Environment are rendered only when the model is not training. 
During training, the runs directory will contain the graphs of the Value-Function
error and the Reward collected by the agent, to be visualized with tensorboard.

To train on a different environment create a .json file in the models directory
containing the required training parameters:

| Parameter | Description |
|-----------|-------------|
|LR| learning rate|
|GAMMA| Discount Factor|
|EPOCHS| N. of epochs to perform on each batch of timesteps |
|BATCHSIZE| Mini-batch size for training over timesteps |
|TRAJECTORIES| N. of trajectories to collect before performing a training session.|
|MAX_STEPS| Maximum steps performed in the environment|
|CHECKPOINT| N. of full model training before saving a checkpoint |
|EPISODES| N. of full environment episodes performed |
|STD| Standard Deviation to be applied to random actions chosen by the agent.|
|NETSIZE| N. of hidden nodes. This is a simple 3-layer Feed-Forward network and the default 64 is good for simple environments.|
|C2| Entropy coefficient for PPO.|

You may just copy one among the ones in the models directory and tweak as needed.

