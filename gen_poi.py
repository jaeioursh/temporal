import numpy as np
pos=[]
val=[]
for i in range(-1,4):
    for j in range(-1,4):
        if not(i==1 and j==1):
            pos.append([i*0.5,j*0.5])
            val.append(np.random.randint(1,11)*0.1)
print(pos)
print(val)