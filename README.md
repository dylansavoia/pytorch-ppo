# Proximal Policy Optimization
A py-torch implementation of the PPO algorithm.

<div align="center"><img src=gifs/InvDoublePend.gif /></div>
<br>

The code is made of two files:
- ppo_agent.py which implements the PPO algorithm itself,
- main.py which renders the chosen environment and runs the agent on it.

Inside the /models directory there are pre-trained models for demo, but any 
**continuous action space** environment could be used (with varying degrees of success).


## Install
Install all dependencies using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):
```bash
$ conda env create --file environment.yml
```

Then, to run an environment, the relative platform and its dependencies must be installed as well.

E.g. to install Box2d or Mujoco run:
```bash
$ pip install gymnasium[box2d]
$ pip install gymnasium[mujoco]
```

## Train and Run Environments
It is possible to run the code with the following command:
```bash
$ python main.py -h
usage: main.py [-h] [--train] environment

Train PPO models and run Gym environments

positional arguments:
  environment  MountainCarContinuous-v0, Pendulum-v1, InvertedPendulum-v4,
               InvertedDoublePendulum-v4

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

## Show Tensorboard plots
During training, a _runs/_ directory is created containing the logs for tensorboard.

To open a new tensorboard session with the logs run:

```bash
$ tensorboard --logdir=runs
TensorFlow installation not found - running with reduced feature set.

NOTE: Using experimental fast data loading logic. To disable, pass
    "--load_fast=false" and report issues on GitHub. More details:
    https://github.com/tensorflow/tensorboard/issues/4784

Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.12.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

Then you can navigate to the localhost address at the given port, to show the log
