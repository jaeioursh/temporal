from train import make_env,train
from rewards.align import align
from rewards.attention import attention
import multiprocessing as mp
import numpy as np




def experiment(n_agents,reward_type,trial):
    fname="-".join([str(i) for i in [n_agents,reward_type,trial]])+".npy"
    env=make_env(n_agents)
    
    if reward_type==0:
        reward_mechanism=align(n_agents)
    elif reward_type==1:
        reward_mechanism=attention(n_agents)
    R=train(env,reward_mechanism,2)
    np.save("saves/"+fname,np.array(R))

experiment(2,0,0)