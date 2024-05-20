import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
'''
scp -J cookjos@access.engr.oregonstate.edu cookjos@graf200-17.engr.oregonstate.edu:credit/save/* save/
'''
num=8
typ=1
#fig = plt.figure()
specs=[
        [{"type": "scatter3d"}],
    ]
fig = make_subplots(rows=1, cols=1,specs=specs)



for i in [1]:

    
    x,g=[],[]
    for idx in range(1):
        try:
            with open("0-82.pkl","rb") as f:
                X,G=pickle.load(f)
        except:
            pass
        x.append(X)
        g.append(G)

    x,g=np.array(x)[0],np.array(g)[0]
    VMAX=np.max(g)
    print(-np.min(g))
    
    idx=(-g).argsort()
    
    print(idx.shape,x.shape)
    x,g=x[idx,:],g[idx]
    for qq in range(6):
        for xx in x[qq]:
            print(np.round(xx,6),sep=", ",end=" ")
        print(g[qq])

    x=np.array(x)
    lr, hidden, buffer = x.T

    
    barloc=[1/3.0,2.0/3.0,1]
    
    fig.add_trace(go.Scatter3d(
        x=lr,
        y=hidden,
        z=buffer,
        mode='markers',
        marker=dict(
            cmin=0.0,
            cmax=VMAX*1.0,
            size=6,#np.sqrt(buffer/10),
            color=g,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8,
            colorbar=dict(title = "G",x=barloc[i-1])
        )
    ),row=1,col=i)

    g=np.array([g]).T
    
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(x)
    reg = LinearRegression().fit(X, g)
    print(reg.coef_)
    '''
    ax = fig.add_subplot(130+i,projection='3d')

    p=ax.scatter(lr, hidden, batch, s=np.sqrt(buffer/10),c=-g, vmin=0,vmax=VMAX, marker='o')

    ax.set_xlabel('lr')
    ax.set_ylabel('hidden')
    ax.set_zlabel('batch')
    plt.colorbar(p)
    plt.title(str(num)+" agents")
    print(x.shape,g.shape)
    '''

# tight layout
#fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
lbls=dict(
xaxis_title='lr',
yaxis_title='batch',
zaxis_title='buffer')
fig.update_layout(scene =lbls,scene2 =lbls,scene3=lbls )
fig.show()
#plt.show()