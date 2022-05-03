import vlm
import json
from matplotlib import projections, pyplot as plt
import numpy as np

test=vlm.VLM()

with open("plane1.json",'r') as f:
    example=json.load(f)

n_range=(1,15)
m_range=(1,15)

Q_inf=1
alpha=10
beta=0

results=[]
for n in range(n_range[0],n_range[1]+1):
    for m in range(m_range[0],m_range[1]+1):
        mesh=example
        mesh["plane"]["wing"][0]["n"]=n
        mesh["plane"]["wing"][0]["section"][0]["m"]=m

        test.geometry(data=mesh)
        L,Di=test.vlm(Q_inf_mod=Q_inf,alpha=alpha,beta=beta)

        results.append((n,m,L,Di))

n=np.array([result[0] for result in results])
m=np.array([result[1] for result in results])
L=np.array([result[2] for result in results])
Di=np.array([result[3] for result in results])

#n,m=np.meshgrid(n,m)

fig=plt.figure(1)
ax=plt.axes(projection='3d')

ax.plot_trisurf(n,m,L)
ax.set_xlabel('n')
ax.set_ylabel('m')
ax.set_zlabel('L (N)')

fig=plt.figure(2)
ax=plt.axes(projection='3d')

ax.plot_trisurf(n,m,Di)
ax.set_xlabel('n')
ax.set_ylabel('m')
ax.set_zlabel('Di (N)')

plt.show()