import numpy as np
import matplotlib.pyplot as plt
pos=[]
val=[]
q=0
for i in range(-1,4):
    for j in range(-1,4):
        q+=1
        if not(i==1 and j==1) and q%2==1:
            pos.append([i*0.5,j*0.5])
            val.append(np.random.randint(1,11)*0.1)


print(pos)
print(val)

pos=np.array(pos)
plt.scatter(pos[:,0],pos[:,1])
plt.show()