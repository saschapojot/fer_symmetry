import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
#this script loads M2, with rescalings
# for all N
if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]

T_lower=0.8
T_upper=1.125
Tc=1.13
def load_T_M2_for_one_N(N):
    csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
    inCsvFile=csvDataFolderRoot+"/magnetization_plot.csv"
    df=pd.read_csv(inCsvFile)
    TVec=np.array(df["T"])
    MValsAll=np.array(df["M"])
    M2ValsAll=MValsAll**2

    mask = (TVec > T_lower) & (TVec < T_upper)
    TInds = np.where(mask)[0]
    TToPlt=TVec[TInds]
    M2_to_plot=M2ValsAll[TInds]
    return TToPlt, M2_to_plot

NVec=[50,100,150,200]

T_vecs_all=[]#corresponding to each N
M2_vec_all=[]#corresponding to each N

for N in NVec:
    TToPlt, M2_to_plot=load_T_M2_for_one_N(N)
    T_vecs_all.append(TToPlt)
    M2_vec_all.append(M2_to_plot)

def T_M_to_rescaled(TToPlt_one_N,M2_to_plot_one_N,N,beta,nu):
    TToPlt_one_N=np.array(TToPlt_one_N)
    M2_to_plot_one_N=np.array(M2_to_plot_one_N)

    eps_one_N=Tc/np.abs(TToPlt_one_N-Tc)*N**(-1/nu)
    tau_one_N=np.abs(TToPlt_one_N-Tc)/Tc
    M2_rescaled_one_N=N**(2*beta/nu)*M2_to_plot_one_N*eps_one_N**(2*beta)

    return eps_one_N,M2_rescaled_one_N,tau_one_N


beta=0.125
nu=1

plt.figure()
for ind,N in enumerate(NVec):
    print(f"ind={ind}, N={N}")
    TToPlt_one_N=T_vecs_all[ind]
    M2_to_plot_one_N=M2_vec_all[ind]

    eps_one_N,M2_rescaled_one_N,tau_one_N=T_M_to_rescaled(TToPlt_one_N,M2_to_plot_one_N,N,beta,nu)
    plt.scatter(eps_one_N,M2_rescaled_one_N,label=f"N={N}")
    print(f"tau_one_N={tau_one_N}")

plt.xlim(0,0.1)

plt.xlabel(r"$\epsilon$")
plt.ylabel(r"")
plt.legend(loc="best")
outDir=f"../dataAll/row{row}/"
plt.savefig(outDir+"/M2_rescale.png")
plt.close()