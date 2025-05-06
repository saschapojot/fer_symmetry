import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
import scipy.stats as stats

#this script loads Binder ratio for all N
#this script also computes the intersections of Binder ratio between different N
if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]
N_vec=[32,64]
csvDataFolderRoot_vec=[f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/" for N in N_vec]
csv_out_dir=f"../dataAll/row{row}/"
Path(csv_out_dir).mkdir(exist_ok=True,parents=True)
fig,ax=plt.subplots()
for j in range(0,len(N_vec)):
    N=N_vec[j]
    inCsvFile=csvDataFolderRoot_vec[j]+"/magnetization_plot.csv"
    df=pd.read_csv(inCsvFile)
    TVec=np.array(df["T"])
    U_L=np.array(df["U_L"])
    mask = (TVec > 11) & (TVec<11.5)
    TInds = np.where(mask)[0]
    TToPlt=TVec[TInds]
    ax.scatter(TToPlt,U_L[TInds],label=f"{N}")

ax.set_xlabel('$T$')
ax.set_ylabel("$U_{L}$")
plt.legend(loc="best")
plt.savefig(csv_out_dir+"/UL_all_N.png")
plt.close()


N0_ind=0#32
N1_ind=1# 64

#N0_ind
inCsvFile0=csvDataFolderRoot_vec[N0_ind]+"/magnetization_plot.csv"
df0=pd.read_csv(inCsvFile0)
TVec0=np.array(df0["T"])
U_L0=np.array(df0["U_L"])
mask0=(TVec0>11.25) &( TVec0<11.44)
TInds0 = np.where(mask0)[0]
U_L0_binder=U_L0[TInds0]
T_vec0_binder=TVec0[TInds0]
print(f"T_vec0_binder={T_vec0_binder}")
print(f"U_L0_binder={U_L0_binder}")

#N1_ind
inCsvFile1=csvDataFolderRoot_vec[N1_ind]+"/magnetization_plot.csv"
df1=pd.read_csv(inCsvFile1)
TVec1=np.array(df1["T"])
U_L1=np.array(df1["U_L"])
mask1=(TVec1>11.25) &( TVec1<11.44)
TInds1 = np.where(mask1)[0]
T_vec1_binder=TVec1[TInds1]
U_L1_binder=U_L1[TInds1]
print(f"T_vec1_binder={T_vec1_binder}")
print(f"U_L1_binder={U_L1_binder}")

#last 2 data points for N0
x1=T_vec0_binder[0]
x2=T_vec0_binder[1]
print(f"x1={x1}")
print(f"x2={x2}")
y1=U_L0_binder[0]
y2=U_L0_binder[1]
print(f"y1={y1}")
print(f"y2={y2}")

#last 2 data points for N1
x3=T_vec1_binder[0]
x4=T_vec1_binder[1]
y3=U_L1_binder[0]
y4=U_L1_binder[1]
print(f"x3={x3}")
print(f"x4={x4}")
print(f"y3={y3}")
print(f"y4={y4}")

A=np.zeros((2,2))
b=np.array([x1*(y2-y1)/(x2-x1)-y1, x3*(y4-y3)/(x4-x3)-y3])
A[0,0]=(y2-y1)/(x2-x1)
A[0,1]=-1

A[1,0]=(y4-y3)/(x4-x3)

A[1,1]=-1
solution_0_1=np.linalg.solve(A,b)
print(solution_0_1)
