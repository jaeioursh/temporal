import numpy as np

import pyximport
pyximport.install() 

from rover_domain_core_gym import RoverDomainGym
from code.ccea_2 import *
from code.mod import assignDifferenceRewardTemporal


#pri alignment multiagent tumernt(vals)
def make_env(nagents,coupling=1,rand=0):
    vals =np.array([0.8,1.0,0.6,0.3,0.2,0.1])
    
    pos=np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.5],
        [0.0, 0.5],
        [1.0, 0.0]
    ])

    sim = RoverDomainGym(nagents,30,pos,vals)
 
    sim.data["Coupling"]=coupling
    sim.data['Number of Agents']=nagents
    sim.data['Trains per Episode']=32 #pop size

    sim.data["Reward Function"] = assignDifferenceRewardTemporal
    sim.data["Evaluation Function"] = assignDifferenceRewardTemporal

    obs=sim.reset()
    
    return sim


#pri alignment multiagent tumernt(vals)
def make_env2(nagents,coupling=1,rand=0):
    vals=[0.30000000000000004, 0.1, 0.4, 0.5, 0.30000000000000004, 0.9, 0.8, 1.0, 0.30000000000000004, 0.7000000000000001, 0.9, 0.8, 1.0, 0.30000000000000004, 0.5, 0.1, 0.4, 0.7000000000000001, 0.9, 0.4, 0.9, 0.6000000000000001, 0.4, 0.6000000000000001]
    vals=np.array(vals)
    pos=np.array([[-0.5, -0.5], [-0.5, 0.0], [-0.5, 0.5], [-0.5, 1.0], [-0.5, 1.5], [0.0, -0.5],
                      [0.0, 0.0], [0.0, 0.5], [0.0, 1.0], [0.0, 1.5], [0.5, -0.5], [0.5, 0.0], [0.5, 1.0],
                      [0.5, 1.5], [1.0, -0.5], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, 1.5], [1.5, -0.5],
                      [1.5, 0.0], [1.5, 0.5], [1.5, 1.0], [1.5, 1.5]])

    sim = RoverDomainGym(nagents,40,pos,vals)
 
    sim.data["Coupling"]=coupling
    sim.data['Number of Agents']=nagents
    sim.data['Trains per Episode']=32 #pop size

    sim.data["Reward Function"] = assignDifferenceRewardTemporal
    sim.data["Evaluation Function"] = assignDifferenceRewardTemporal

    obs=sim.reset()
    
    return sim




def train(env, reward_mechanism, generations=4000):
    R=[]
    pos=[]
    initCcea(input_shape=8, num_outputs=2, num_units=20)(env.data)

    populationSize=len(env.data['Agent Populations'][0])
    pop=env.data['Agent Populations']
    
    nagents=env.data['Number of Agents']

    for gen in range(generations):
        pos=[]
        Globals=[]
        evalutaion_data=[]
        for worldIndex in range(populationSize):
            
            env.data["World Index"]=worldIndex
            state = env.reset() 
            p=[np.array(env.data["Agent Positions"])]
            done=False 
            assignCceaPolicies(env.data)
            trajectories=[[state[a]] for a in range(nagents)]
            policyCol=env.data["Agent Policies"]
            while not done:
                action=[]
                for s,pol in zip(state,policyCol):
        
                    a = pol.get_action(s)
                    action.append(a)
                action = np.array(action)*2.0
                state, D, done, info = env.step(action)
                for a in range(nagents):
                    trajectories[a].append(state[a])
                p.append(np.array(env.data["Agent Positions"]))
            pos.append(p)
            for a in range(nagents):
                trajectories[a]=np.array(trajectories[a])

            G=env.data["Global Reward"]
            D=env.data["Agent Rewards"]
            Globals.append(G)
            if reward_mechanism != "g" and reward_mechanism != "d":
                for a in range(nagents):
                    evalutaion_data.append([trajectories[a],policyCol[a],a])
                    reward_mechanism.add(trajectories[a],G,a)
            else:
                if reward_mechanism == "g":
                    for pol in policyCol:
                        pol.fitness=G
                if reward_mechanism == "d":
                    for d,pol in zip(D,policyCol):
                        pol.fitness=d
        R.append(max(Globals))
        pos=pos[np.argmax(Globals)]
        if reward_mechanism != "g" and reward_mechanism != "d":
            reward_mechanism.train()
            for trajectory,policy,agent_index in evalutaion_data:
                policy.fitness=reward_mechanism.evaluate(trajectory,agent_index)
        print("Gen :"+str(gen)+"  Best G: "+str(max(Globals)))

        evolveCceaPolicies(env.data)
    return R,pos

if __name__=="__main__":
    env=make_env2(4,2)
    train(env,"d")