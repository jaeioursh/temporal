from train import make_env,train
from rewards.align import align
from rewards.attention import attention
from rewards.fitnesscritic import fitnesscritic
import multiprocessing as mp
import numpy as np
import pickle as pkl
import time


def experiment(n_agents,reward_type,trial):
    fname="-".join([str(i) for i in [n_agents,reward_type,trial]])+".pkl"
    env=make_env(n_agents)
    
    if reward_type==0:
        reward_mechanism=align(n_agents,loss_f=0)
    elif reward_type==1:
        reward_mechanism=align(n_agents,loss_f=2) #alignment
    elif reward_type==2:
        reward_mechanism=attention(n_agents,loss_f=0)
    elif reward_type==3:
        reward_mechanism=attention(n_agents,loss_f=2) #alignment
    elif reward_type==4:
        reward_mechanism=fitnesscritic(n_agents)

    R,pos=train(env,reward_mechanism)
    with open("saves/"+fname,"wb") as f:
        pkl.dump([R,pos],f)

    
for n_agents in range(4,6,8):
    for reward_type in [0,1,2,3,4]:
        procs=[]
        for trial in range(12):
            p=mp.Process(target=experiment,args=(n_agents,reward_type,trial))
            time.sleep(0.05)
            procs.append(p)
            #p.join()git ad
        for p in procs:
            p.join()