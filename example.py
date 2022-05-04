import vlm
import numpy as np
from matplotlib import projections, pyplot as plt

example=vlm.VLM()
example.geometry("plane1.json")

Q_inf=30

alpha=np.linspace(0,10,10)
beta=0

L_list=[]
Di_list=[]
for a in alpha:
    L,D=example.vlm(Q_inf_mod=Q_inf,alpha=a,beta=beta)

    L_list.append(L)
    Di_list.append(D)

#print(f"Qinf={Q_inf} m/s, alpha={alpha} deg, beta={beta} deg")
#print(f"L={L} N, Di={D} N")

example.plot_mesh(cp=False)

fig1,ax1=plt.subplots()

ax1.plot(alpha,L_list)
ax1.set_xlabel("Alpha (deg)")
ax1.set_ylabel("C_L")

fig2,ax2=plt.subplots()

ax2.plot(alpha,Di_list)
ax2.set_xlabel("Alpha (deg)")
ax2.set_ylabel("C_Di")

plt.show()
