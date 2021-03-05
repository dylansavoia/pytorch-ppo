import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
import numpy as np
import sys
from torch.utils.tensorboard import SummaryWriter

glob_i = 0
glob_error = 0

class PPOAgent():
    def __init__(self, TRAIN, env=None, traj=3, net_size=64, net_std=1,
                lr=1e-4, bs=100, y=0.99, ep=10, W=None,
                c1=0.5, c2=0):
        self.TRAIN  = TRAIN
        self.LR     = lr
        self.GAMMA  = y
        self.BATCHSIZE  = bs
        self.EPOCHS = ep
        self.NTRAJ = traj
        self.env = env
        self.C1 = c1
        self.C2 = c2

        self.WRITER = W

        inp = env.observation_space.shape[0]
        out = env.action_space.shape[0]
        self.model = AgentNet(inp, out, h=net_size, std=net_std)
        self.opt = optim.Adam(self.model.parameters(), lr=self.LR)

        self.trajectories = []
        self.states   = []
        self.rewards  = []
        self.actions  = []
        self.values   = []
        self.logAprob = []

    def __call__(self, state):
        # Each time an action is required, we save the
        # state value for computing advantages later
        state = torch.FloatTensor(state)
        normal, v = self.model(state) 
        action = normal.sample()

        self.values.append(v.unsqueeze(0))
        self.actions.append(action)
        self.logAprob.append(normal.log_prob(action))

        return action

    def observe(self, s, r, s1, done, NEPISODE):
        if not self.TRAIN: return

        s  = torch.FloatTensor(s).unsqueeze(0)
        s1 = torch.FloatTensor(s1).unsqueeze(0)

        self.states.append(s)

        # For rewards we only maintain the value.
        self.rewards.append(r)

        if done:
            Bufs = torch.cat(self.states).detach()
            Bufa = torch.cat(self.actions).detach()
            Bufv = torch.cat(self.values).detach()
            Bufp = torch.cat(self.logAprob).detach()
            Bufr = self.rewards
            self.trajectories.append({
                "S": Bufs,
                "A": Bufa,
                "r": Bufr,
                "V": Bufv,
                "LogP": Bufp
            })

            self.states = []
            self.actions = []
            self.logAprob = []
            self.values = []
            self.rewards = []

            # Update Condition
            if NEPISODE % self.NTRAJ == 0: 
                self.update(s1, done) 

    def update(self, s1, done):
        EPS = 0.2

        # Compute Discounted Rewards
        for tr in self.trajectories:
            g = []
            tot = 0
            for reward in tr['r'][::-1]:
                tot = reward + self.GAMMA*tot
                g.insert(0, torch.FloatTensor([tot]))

            G = torch.cat(g).unsqueeze(1).detach()
            Adv = G - tr['V']
            Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-8)
            tr['G'] = G
            tr['Adv'] = Adv

        # At this point each item in trajectory will contain:
        # S, A, r, V, LogP, G, Adv
        S = torch.cat([x['S'] for x in self.trajectories], 0)
        A = torch.cat([x['A'] for x in self.trajectories], 0)
        Adv = torch.cat([x['Adv'] for x in self.trajectories], 0)
        Log_old = torch.cat([x['LogP'] for x in self.trajectories], 0)
        G = torch.cat([x['G'] for x in self.trajectories], 0)
        V = torch.cat([x['V'] for x in self.trajectories], 0)

        global glob_i
        global glob_error
        glob_error = 0

        bufsize = S.size(0)

        for ep in range(self.EPOCHS*(bufsize//self.BATCHSIZE+1)):
            ids = np.random.randint(0, bufsize,
                    min(self.BATCHSIZE, bufsize))

            bS, bA, bAdv = S[ids,:], A[ids,:], Adv[ids,:]

            normal, Vnew = self.model(bS)
            logAprob_old = Log_old[ids,:]
            logAprob = normal.log_prob(bA)

            # L_CLIP
            ratio = (logAprob - logAprob_old).exp().squeeze(0)
            m1 = ratio * bAdv
            m2 = torch.clamp(ratio, 1.0 - EPS, 1.0 + EPS) * bAdv
            L_CLIP = torch.min(m1, m2).mean()

            # L_VF
            L_VF = (G[ids,:] - Vnew).pow(2).mean()
            glob_error += L_VF

            # Entropy
            E = normal.entropy().mean()

            # Total Loss
            # print(self.C1, self.C2)
            L = -L_CLIP + self.C1 * L_VF - self.C2 * E

            # Apply Gradients
            self.opt.zero_grad()
            L.backward()
            self.opt.step()

        # Update Graphs
        self.WRITER.add_scalar("L_VF", L_VF, glob_i)
        glob_i += 1
        self.trajectories = []

    def load(self, path):
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
        except FileNotFoundError:
            print(f'Error: {path} not found.')
            exit()


    def save(self, path):
        torch.save(self.model.state_dict(), path)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class AgentNet(nn.Module):
    def __init__(self, inp, out, h=32, std=0):
        super(AgentNet, self).__init__()
        
        self.policy = nn.Sequential(
            nn.Linear(inp, h),
            nn.Tanh(),
            nn.Linear(h, h//2),
            nn.Tanh(),
            nn.Linear(h//2, out),
        )
        self.log_std = nn.Parameter(torch.ones(1, out) * std)
        
        self.value = nn.Sequential(
            nn.Linear(inp, h),
            nn.Tanh(),
            nn.Linear(h, h//2),
            nn.Tanh(),
            nn.Linear(h//2, 1)
        )
        

    def forward(self, x):
        value = self.value(x)
        mu    = self.policy(x).unsqueeze(0)
        # print(self.log_std)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value


