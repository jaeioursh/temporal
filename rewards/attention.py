import numpy as np
import torch
from collections import deque
from random import sample
import math
import torch.nn as nn

def scaled_dot_product_attention(query, key, value,device) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L,D = query.size(-2), key.size(-1)
    scale_factor = 1 / math.sqrt(D)
    attn_bias = torch.zeros(L, L, dtype=query.dtype).to(device)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value,attn_weight

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

class Net(nn.Module):
    def __init__(self,device,hidden=20*4,lr=5e-4,seq_len=46,idim=8,loss_fn=0):#*4
        super(Net, self).__init__()
        learning_rate=lr
        self.device=device

        
        self.w1=nn.Linear(idim,hidden)
        
        self.wout1=nn.Linear(hidden,hidden)
        self.wout2=nn.Linear(hidden,1)
        self.acti=nn.LeakyReLU()
        self.pos=getPositionEncoding(seq_len,idim)
        self.pos=torch.from_numpy(self.pos.astype(np.float32)).to(self.device)

        if loss_fn==0:
            self.loss_fn = lambda x,y: torch.nn.MSELoss(reduction='sum')(x[:,-1,:],y[:,-1,:])
        elif loss_fn==1:
            self.loss_fn = self.alignment_loss
        elif loss_fn ==2:
            self.loss_fn = lambda x,y: self.alignment_loss(x[:,-1,:],y[:,-1,:]) + torch.nn.MSELoss(reduction='sum')(x[:,-1,:],y[:,-1,:])

        self.sig = torch.nn.Sigmoid()

        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self,x):
        
        KQV=self.w1(x+self.pos)
        res,attn=scaled_dot_product_attention(KQV,KQV,KQV,self.device)
        return self.wout2(self.acti(self.wout1(res)))

    def feed(self,x):
        x=torch.from_numpy(x.astype(np.float32)).to(self.device)
        pred=self.forward(x)
        return pred.cpu().detach().numpy()
        
    
    def train(self,x,y,verb=0):
        x=torch.from_numpy(x.astype(np.float32)).to(self.device)
        y=torch.from_numpy(y.astype(np.float32)).to(self.device)
        pred=self.forward(x)
        
        loss=self.loss_fn(pred,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu().detach().item()

    def alignment_loss(self,o, t):
        o=o
        t=t
        ot=torch.transpose(o,0,1)
        tt=torch.transpose(t,0,1)

        O=o-ot
        T=t-tt

        align = torch.mul(O,T)
        #print(align)
        align = self.sig(align)
        loss = -torch.mean(align)
        return loss
    
class attention():
    def __init__(self,nagents,device,loss_f=0,params=None):
        self.nagents=nagents
        if params is None:
            self.nets=[Net(device,loss_fn=loss_f).to(device) for i in range(nagents)]
            self.hist=[deque(maxlen=30000) for i in range(nagents)]
        else:
            lr,hidden,hist=params
            hidden,hist=int(hidden),int(hist)
            self.nets=[Net(device,hidden=hidden,lr=lr,loss_fn=loss_f).to(device) for i in range(nagents)]
            self.hist=[deque(maxlen=hist) for i in range(nagents)]

    def add(self,trajectory,G,agent_index):
        self.hist[agent_index].append([trajectory,G])

    def evaluate(self,trajectory,agent_index):
        return self.nets[agent_index].feed(trajectory)[-1][0]

    def train(self):
        for a in range(self.nagents):
            for i in range(100):
                if len(self.hist[a])<24:
                    trajG=self.hist[a]
                else:
                    trajG=sample(self.hist[a],24)
                S,G=[],[]
                for traj,g in trajG:
                    S.append(traj)
                    G.append([[0.0]]*(len(traj)-1)+[[g]])
                S,G=np.array(S),np.array(G)
                self.nets[a].train(S,G)
[[0.0]]
if __name__ == "__main__":
    L=10
    net=Net(idim=6,seq_len=L,loss_fn=0)
    s=np.arange(-10,10,0.5)
    n=len(s)
    X=np.zeros((n,L,6))
    Y=np.zeros((n,L,1))
    X[:,0,0]=s
    Y[:,-1,0]=s
    #print(X,Y)
    for i in range(10000):
        l=net.train(X,Y)
        
        if i%500==0:
            print(i,l)
            print(net.feed(X[0,:,:]))