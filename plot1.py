import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
nagents=4
window=20
lbl=["End State","End State Aln","Attn.","Attn Aln","Fit Critic"]
for exp in range(5):
    R=[]
    for trial in range(20,30):
        fname="-".join([str(s) for s in [nagents,exp,trial]])+".pkl"
        print(fname)
        with open("saves/"+fname,"rb") as f:
            r,pos=pkl.load(f)
            r=np.array(r)
            r=np.average(r.reshape(-1, window), axis=1)
            R.append(r)
    
    R=np.array(R)
    print(R.shape)
    mu=np.mean(R,axis=0)
    std=np.std(R,axis=0)/np.sqrt(R.shape[0])
   
    X=[i*window for i in range(len(mu))]

    
    plt.fill_between(X,mu-std,mu+std,alpha=0.2, label='_nolegend_')

    plt.plot(X,mu,label=lbl[exp])
plt.legend()
plt.title(str(nagents)+" agents")
plt.xlabel("Generations")
plt.ylabel("Global Evaluation")
plt.savefig("figs/1_"+str(nagents)+".75.png")
#plt.show()