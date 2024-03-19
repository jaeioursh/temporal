import numpy as np
import torch
from collections import deque
from random import sample

class Net():
    def __init__(self,hidden=20*4,lr=5e-3,loss_fn=2):#*4
        learning_rate=lr
        

        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(8, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden,1)
        )
        
            
        if loss_fn==0:
            self.loss_fn = torch.nn.MSELoss(reduction='sum')
        elif loss_fn==1:
            self.loss_fn = self.alignment_loss
        elif loss_fn ==2:
            self.loss_fn = lambda x,y: self.alignment_loss(x,y) + torch.nn.MSELoss(reduction='sum')(x,y)

        self.sig = torch.nn.Sigmoid()

        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    def feed(self,x):
        x=torch.from_numpy(x.astype(np.float32))
        pred=self.model(x)
        return pred.detach().numpy()
        
    
    def train(self,x,y,shaping=False,n=5,verb=0):
        x=torch.from_numpy(x.astype(np.float32))
        y=torch.from_numpy(y.astype(np.float32))
        pred=self.model(x)
        
        loss=self.loss_fn(pred,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()

    def alignment_loss(self,o, t,shaping=False):
        if shaping:
            o=o+t
        ot=torch.transpose(o,0,1)
        tt=torch.transpose(t,0,1)

        O=o-ot
        T=t-tt

        align = torch.mul(O,T)
        #print(align)
        align = self.sig(align)
        loss = -torch.mean(align)
        return loss
    
class align():
    def __init__(self,nagents,loss_f=0):
        self.nagents=nagents
        self.nets=[Net(loss_fn=loss_f) for i in range(nagents)]
        self.hist=[deque(maxlen=30000) for i in range(nagents)]

    def add(self,trajectory,G,agent_index):
        self.hist[agent_index].append([trajectory,G])

    def evaluate(self,trajectory,agent_index):
        return self.nets[agent_index].feed(trajectory[-1])

    def train(self):
        for a in range(self.nagents):
            for i in range(100):
                if len(self.hist[a])<32:
                    trajG=self.hist[a]
                else:
                    trajG=sample(self.hist[a],32)
                S,G=[],[]
                for traj,g in trajG:
                    S.append(traj[-1])
                    G.append([g])
                S,G=np.array(S),np.array(G)
                self.nets[a].train(S,G)

