import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def scaled_dot_product_attention(query, key, value) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L,D = query.size(-2), key.size(-1)
    scale_factor = 1 / math.sqrt(D)
    attn_bias = torch.zeros(L, L, dtype=query.dtype)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value,attn_weight

class net(nn.Module):
    def __init__(self,idim=1,odim=1,hidden=20):
        super(net, self).__init__()

        self.wK=nn.Linear(idim,hidden)
        self.wQ=nn.Linear(idim,hidden)
        self.wV=nn.Linear(idim,hidden)
        self.wout1=nn.Linear(hidden,hidden)
        self.wout2=nn.Linear(hidden,odim)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        lr=1e-2
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.acti=nn.ReLU()

    def forward(self,x,attn_idx=0):
        K=self.wK(x)
        Q=self.wQ(x)
        V=self.wV(x)
        
        res,attn=scaled_dot_product_attention(Q,K,V)
        
        #res=F.scaled_dot_product_attention(Q,K,V)
        #attn=res
        if not attn_idx:
            return self.wout2(res)
            #return self.wout2(self.acti(self.wout1(res)))
        else:
            return attn
        
    
    def feed(self,x,attn_idx=0):
        x=torch.from_numpy(x.astype(np.float32))
        pred=self.forward(x,attn_idx)
        return pred.detach().numpy()
        
    
    def train(self,x,y):
        x=torch.from_numpy(x.astype(np.float32))
        y=torch.from_numpy(y.astype(np.float32))
        pred=self.forward(x)
        loss=self.loss_fn(pred,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
    

atn=net(1,1,2)

x=[[[1],[0],[0],[0]],
   [[0],[1],[0],[0]],
   [[0],[0],[1],[0]],
   [[0],[0],[0],[1]]]

y=[[[0],[1],[0],[0]],
   [[0],[0],[1],[0]],
   [[0],[0],[0],[1]],
   [[1],[0],[0],[0]]]
x,y=[x[0]],[y[0]]
x,y=np.array(x),np.array(y)

for i in range(3001):
    print(atn.train(x,y))

    if i%1000==0:
        print(np.round(atn.feed(x),3))
        print(atn.feed(x[0],1))