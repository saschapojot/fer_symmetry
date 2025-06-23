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
#this script loads M, with rescalings
# for all N
if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]

T_lower=1
T_upper=1.15
J_abs=1/2
Tc_exact=2*J_abs/np.log(1+np.sqrt(2))
Tc=1.134#Tc_exact
def load_T_M_for_one_N(N):
    csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
    inCsvFile=csvDataFolderRoot+"/magnetization_plot.csv"
    df=pd.read_csv(inCsvFile)
    TVec=np.array(df["T"])
    MValsAll=np.array(df["M"])
    # M2ValsAll=MValsAll**2

    mask = (TVec > T_lower) & (TVec < T_upper)
    TInds = np.where(mask)[0]
    TToPlt=TVec[TInds]
    M_to_plot=MValsAll[TInds]
    return TToPlt, M_to_plot

NVec=[5,10,20,30,40,50,60,70,80,90]

T_vecs_all=[]#corresponding to each N
M_vec_all=[]#corresponding to each N

for N in NVec:
    TToPlt, M_to_plot=load_T_M_for_one_N(N)
    T_vecs_all.append(TToPlt)
    M_vec_all.append(M_to_plot)

def T_M_to_rescaled(TToPlt_one_N,M_to_plot_one_N,N,beta,nu):
    TToPlt_one_N=np.array(TToPlt_one_N)
    M_to_plot_one_N=np.array(M_to_plot_one_N)


    tau_one_N=(TToPlt_one_N-Tc)/Tc*N**(1/nu)
    M_rescaled_one_N=M_to_plot_one_N*N**(beta/nu)

    return M_rescaled_one_N,tau_one_N


beta=0.125
nu=1.

plt.figure()
for ind,N in enumerate(NVec):
    print(f"ind={ind}, N={N}")
    TToPlt_one_N=T_vecs_all[ind]
    M_to_plot_one_N=M_vec_all[ind]

    M_rescaled_one_N,tau_one_N=T_M_to_rescaled(TToPlt_one_N,M_to_plot_one_N,N,beta,nu)
    plt.scatter(tau_one_N,M_rescaled_one_N,label=f"N={N}")
    print(f"tau_one_N={tau_one_N}")

# plt.xlim(0,0.1)

plt.xlabel(r"$\tau N^{\frac{1}{\nu}}$")
plt.ylabel(r"$M N^{\frac{\beta}{\nu}}$")
plt.legend(loc="best")
outDir=f"../dataAll/row{row}/"
plt.savefig(outDir+"/M_rescale.png")
plt.close()

plt.figure()
for ind,N in enumerate(NVec):
    print(f"ind={ind}, N={N}")
    TToPlt_one_N=T_vecs_all[ind]
    M_to_plot_one_N=M_vec_all[ind]
    plt.scatter(TToPlt_one_N,M_to_plot_one_N,label=f"N={N}")

plt.xlabel(r"$T$")
plt.ylabel(r"$M$")
plt.legend(loc="best")
outDir=f"../dataAll/row{row}/"
plt.savefig(outDir+"/M_compare.png")
plt.close()
