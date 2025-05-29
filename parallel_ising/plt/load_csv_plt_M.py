import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
from scipy.special import ellipk
#This script loads avg U data, with confidence interval
# and plots M, M2 for all T

if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()

N=int(sys.argv[1])
init_path=int(sys.argv[2])
row=sys.argv[3]
csvDataFolderRoot=f"./dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"


inCsvFile=csvDataFolderRoot+"/M_plot.csv"

df=pd.read_csv(inCsvFile)

TVec=np.array(df["T"])
MValsAll=np.array(df["M"])
M2ValsAll=np.array(df["M2"])

mask = (TVec > 0.8) & (TVec <18)
TInds = np.where(mask)[0]
TToPlt=TVec[TInds]
print(TToPlt)
J_abs=1/2

def magnetization_exact(T):
    beta=1/T
    beta_J_2=2*beta*J_abs
    val=np.sinh(beta_J_2)

    return (1-val**(-4))**(1/8)

Tc_guess=1.133
Tc_exact=2*J_abs/np.log(1+np.sqrt(2))
print(f"Tc_exact={Tc_exact}")

T_mag_exact_vec=np.linspace(TToPlt[0],Tc_guess,50)
mag_exact_vec=magnetization_exact(T_mag_exact_vec)
#plt M
fig,ax=plt.subplots()

ax.plot(T_mag_exact_vec,mag_exact_vec,color="magenta",linestyle="-",linewidth=0.5)
plt.scatter(TToPlt,MValsAll[TInds],color="black",label='mc',s=3)
ax.set_xlabel('$T$')
ax.set_ylabel("$|M|$")
ax.set_title("norm of magnetization, unit cell number="+str(N**2))
plt.legend(loc="best")
plt.savefig(csvDataFolderRoot+"/M_abs.png")
plt.close()


#plot M2
fig,ax=plt.subplots()
ax.plot(T_mag_exact_vec,mag_exact_vec**2,color="magenta",linestyle="-",linewidth=0.5)
plt.scatter(TToPlt,M2ValsAll[TInds],color="blue",label='mc',s=3)
ax.set_xlabel('$T$')
ax.set_ylabel("$M^{2}$")
ax.set_title("$M^{2}$, unit cell number="+str(N**2))
plt.legend(loc="best")
plt.savefig(csvDataFolderRoot+"/M2.png")
plt.close()