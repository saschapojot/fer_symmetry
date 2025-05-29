


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
#this script loads M, original values
# for all N
# T>Tc
if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]

T_lower=1.14
T_upper=1.17
plt.figure()

def plt_one_N(N):
    csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
    inCsvFile=csvDataFolderRoot+"/magnetization_plot.csv"
    df=pd.read_csv(inCsvFile)
    TVec=np.array(df["T"])
    MValsAll=np.array(df["M"])
    mask = (TVec > T_lower) & (TVec < T_upper)
    TInds = np.where(mask)[0]
    TToPlt=TVec[TInds]
    M_to_plot=MValsAll[TInds]

    plt.scatter(TToPlt,M_to_plot,label=f"N={N}",s=2)
    return TToPlt,M_to_plot


T_vecs_all=[]
M_vecs_all=[]
plt.xlabel(r"$T$")
plt.ylabel(r"$M$")
NVec=[100,150,200]

for N in NVec:
    TToPlt,M_to_plot= plt_one_N(N)
    T_vecs_all.append(TToPlt)
    M_vecs_all.append(M_to_plot)

plt.legend(loc="best")
# plt.xscale("log")
# plt.yscale("log")
outDir=f"../dataAll/row{row}/"
plt.savefig(outDir+"/mc_M_larger_T.png")
plt.close()

Tc=1.13
beta=7/8
nu=1/1
t_vecs_all=[]
# M_rescaled_vecs_all=[]
plt.figure()
for ind, T_vec_1 in enumerate(T_vecs_all):
    N=NVec[ind]
    t=np.abs(T_vec_1-Tc)/Tc*N**(1/nu)
    M_rescaled=M_vecs_all[ind]*N**(beta/nu)
    plt.scatter(t,M_rescaled,label=f"N={N}",s=5)

plt.legend(loc="best")
# plt.xscale("log")
# plt.yscale("log")
plt.xlabel("$\lambda$")
plt.ylabel("rescaled M")
outDir=f"../dataAll/row{row}/"
plt.savefig(outDir+"/mc_M_rescale_T_larger.png")
plt.close()