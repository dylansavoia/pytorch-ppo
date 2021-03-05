# Proximal Policy Optimization
A py-torch implementation of the PPO algorithm.

The code is made of two files: ppo_agent.py which implements the PPO algo-
rithm itself, and main.py which renders the chosen environment and runs the
agent on it. It is possible to run the code with the following command:

```bash
$ python main.py <env name> <train>
```

The train flag specifies whether training has to occur or not, otherwise the
most recent saved parameters will be used to show performance on the learned
environment. Moreover, this will be rendered only for <train> = False, while
during training, the runs directory will contain the graphs of the Value-Function
error and the Reward collected by the agent, to be visualized with tensorboard.

To train on a different environment create a .json file in the models directory
containing the required training parameters:

- "LR": learning rate
- "GAMMA": Discount Factor
- "BATCHSIZE"
- "EPOCHS"
- "TRAJECTORIES": N. of trajectories to collect before performing a training session.
- "STD": Standard Deviation to be applied to random actions chosen by the agent.
- "NETSIZE": N. of hidden nodes. This is a simple 3-layer Feed-Forward network and the default 64 is good for simple environments.
- "C2": Entropy coefficient for PPO.

You may just copy one among the ones in the models directory and tweak as needed.
All parameters must appear.
