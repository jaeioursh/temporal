#todo: make reward based on full trajectory

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
nagents=4
coupling=2
window=20
exp=2
lbl=["End State","End State Aln","Attn.","Attn Aln","Fit Critic"]


for trial in range(9):
    
    plt.subplot(3,3,trial+1)
    
    fname="-".join([str(s) for s in [nagents,exp,coupling,trial]])+".pkl"
    with open("saves/"+fname,"rb") as f:
        r,pos=pkl.load(f)
    if trial==1:
        plt.title(lbl[exp]+"\n"+str(r[-1]))
    else:
        plt.title(str(r[-1]))
    pos=np.array(pos)
    if 0:
        VALS=[0.8,1.0,0.6,0.3,0.2,0.1]
        poi=np.array([
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.5],
                [0.0, 0.5],
                [1.0, 0.0]
            ])*30
    else:
        
        VALS=[0.30000000000000004, 0.1, 0.4, 0.5, 0.30000000000000004, 0.9, 0.8, 1.0, 0.30000000000000004, 0.7000000000000001, 0.9, 0.8, 1.0, 0.30000000000000004, 0.5, 0.1, 0.4, 0.7000000000000001, 0.9, 0.4, 0.9, 0.6000000000000001, 0.4, 0.6000000000000001]
        poi=np.array([[-0.5, -0.5], [-0.5, 0.0], [-0.5, 0.5], [-0.5, 1.0], [-0.5, 1.5], [0.0, -0.5],
                      [0.0, 0.0], [0.0, 0.5], [0.0, 1.0], [0.0, 1.5], [0.5, -0.5], [0.5, 0.0], [0.5, 1.0],
                      [0.5, 1.5], [1.0, -0.5], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, 1.5], [1.5, -0.5],
                      [1.5, 0.0], [1.5, 0.5], [1.5, 1.0], [1.5, 1.5]])*30
    
    txt=[str(i) for i in VALS]
   
    

    for i in range(nagents):
        mkr='*'
        mkr=[".",",","*","v","^","<",">","1","2","3","4","8"][i]
        mkr="$"+str(i)+"$"
        #clr="k"
        x=pos[1:,i,0]
        y=pos[1:,i,1]
        plt.plot(x,y,color='k',marker=mkr,linewidth=1.0,linestyle=":")

    lgnd=["Agent "+str(i) for i in range(nagents)]
        
    #print(lgnd)

    #plt.legend(lgnd)
    for i in range(len(txt)):
        plt.text(poi[i,0]+1,poi[i,1]+1,txt[i])

    plt.scatter(poi[:,0],poi[:,1],s=100,c='#0000ff',marker="v",zorder=10000)
    #plt.xlim([-5,35])
    #plt.ylim([-5,35])
plt.show()