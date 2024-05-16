from train import make_env2,train
from rewards.align import align
from rewards.attention import attention
from rewards.fitnesscritic import fitnesscritic

from skopt import gp_minimize
from skopt import callbacks
import multiprocessing as mp
import pickle as pkl
import numpy as np

def test(params):
    n_agents=8
    coupling=2
    env=make_env2(n_agents,coupling)
    reward_mechanism=attention(n_agents,"cuda",loss_f=0)
    R,pos=train(env,reward_mechanism,generations=1000)
    return np.mean(R[-10:])

def opt(idx):
    C=[(0.0001, 0.001),(4.0,240.0),(100.0,100000.0)]
    def saver(res):
        with open(str(idx)+"-82.pkl","wb") as f:
            data=[res.x_iters,res.func_vals]
            print("saving "+str(len(res.x_iters)))
            print(data)
            pkl.dump(data,f)
    res = gp_minimize(test, C, n_calls=50,callback=[saver])#,acq_func="PI")
    print(res.x)
    print(res.fun)
    
opt(0)