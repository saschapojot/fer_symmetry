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
#this script loads M2, original values
# for all N


if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]

T_lower=0.8
T_upper=1.2

plt.figure()
def plt_one_N(N):
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

    plt.scatter(TToPlt,M2_to_plot,label=f"N={N}",s=2)
    return TToPlt,M2_to_plot

T_vecs_all=[]
M2_vecs_all=[]



plt.xlabel(r"$T$")
plt.ylabel(r"$M^{2}$")
NVec=[40,50,150,200,250]
for N in NVec:
    TToPlt,M2_to_plot= plt_one_N(N)
    T_vecs_all.append(TToPlt)
    M2_vecs_all.append(M2_to_plot)

plt.legend(loc="best")
# plt.xscale("log")
# plt.yscale("log")
outDir=f"../dataAll/row{row}/"
plt.savefig(outDir+"/mc_M2.png")
plt.close()
J=1/2
Tc_exact=2*J/np.log(1+np.sqrt(2))
Tc=1.13
beta=0.1
nu=1#1/1
t_vecs_all=[]
M_rescaled_vecs_all=[]
plt.figure()
for ind, T_vec_1 in enumerate(T_vecs_all):
    # print(f"ind={ind}, T_vec_1={T_vec_1}")
    N=NVec[ind]
    t=T_vec_1#np.abs(Tc-T_vec_1)/Tc#*N**(1/nu)
    M2_rescaled=M2_vecs_all[ind]*N**(1.2)
    plt.scatter(t,M2_rescaled,label=f"N={N}",s=5)


plt.legend(loc="best")
# plt.xscale("log")
# plt.yscale("log")
plt.xlabel("$\lambda$")
plt.ylabel("rescaled M2")
outDir=f"../dataAll/row{row}/"
plt.savefig(outDir+"/mc_M_rescale.png")
plt.close()