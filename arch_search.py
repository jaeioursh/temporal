import numpy as np
import pickle as pkl
import torch
import torch.utils.data as data_utils
from rewards.attention import Net
def ALIGN(A,B):
    alignment=torch.mul(A-torch.t(A),B-torch.t(B))
    alignment[alignment>0]=1
    alignment[alignment==0]=0.5
    alignment[alignment<0]=0.0
    return torch.mean(alignment)
def MSE(A,B):
    return torch.mean(torch.square(A-B))

def build_dataset():
    for idx in range(4):
        with open("saves/data_"+str(idx)+".pkl","rb") as f:
            state=pkl.load(f)
        nagents=len(state[0][0][0])
        X=[[] for i in range(nagents)]
        Y=[]
        for state,g in state:
            x=[[] for i in range(nagents)]
            for s in state:
                for i in range(nagents):
                    x[i].append(s[i].astype(np.float32))
            for i in range(nagents):
                X[i].append(x[i])    
            Y.append(g)

        X=torch.from_numpy(np.array(X))
        Y=torch.from_numpy(np.array(Y)).float()
        torch.save(X, "saves/X_"+str(idx)+".pt")
        torch.save(Y, "saves/Y_"+str(idx)+".pt")


def main(idx,aidx):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    loss_f=2
    hidden=40
    lr=5e-4
    net=Net(device,hidden=hidden,lr=lr,loss_fn=loss_f).to(device)

    X=torch.load("saves/X_"+str(idx)+".pt").to(device)
    Y=torch.load("saves/Y_"+str(idx)+".pt").to(device)
    print(X.shape,Y.shape)
    batch_size=64*8
    dataset = data_utils.TensorDataset(X[aidx][-50000:], Y.view(len(Y),1)[-50000:])
    data_loader = data_utils.DataLoader(dataset, batch_size, shuffle=True)
    firstround=True
    for epoch in range(500):
        mse=[]
        alg=[]
        for x,y in data_loader:
            if not firstround:
                net.train(x,y)
            pred=net.forward(x)
            mse.append(MSE(pred,y).item())
            alg.append(ALIGN(pred,y).item())
        print(epoch,np.mean(mse),np.mean(alg))
        firstround=False
main(0,0)