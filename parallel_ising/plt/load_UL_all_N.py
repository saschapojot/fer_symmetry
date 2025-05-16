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
N_vec=[32,64,128]
csvDataFolderRoot_vec=[f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/" for N in N_vec]
csv_out_dir=f"../dataAll/row{row}/"
Path(csv_out_dir).mkdir(exist_ok=True,parents=True)
fig,ax=plt.subplots()
for j in range(0,len(N_vec)):
    N=N_vec[j]
    inCsvFile=csvDataFolderRoot_vec[j]+"/magnetization_plot.csv"
    print(f"inCsvFile={inCsvFile}")
    df=pd.read_csv(inCsvFile)
    TVec=np.array(df["T"])
    U_L=np.array(df["U_L"])
    mask = (TVec > 1) & (TVec<1.3)


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
N2_ind=2# 128

#N0_ind
inCsvFile0=csvDataFolderRoot_vec[N0_ind]+"/magnetization_plot.csv"
df0=pd.read_csv(inCsvFile0)
TVec0=np.array(df0["T"])
U_L0=np.array(df0["U_L"])
mask0=(TVec > 1) & (TVec<1.15)
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
mask1=(TVec > 1.11) & (TVec<1.15)
TInds1 = np.where(mask1)[0]
T_vec1_binder=TVec1[TInds1]
U_L1_binder=U_L1[TInds1]
print(f"T_vec1_binder={T_vec1_binder}")
print(f"U_L1_binder={U_L1_binder}")

#N2_ind
inCsvFile2=csvDataFolderRoot_vec[N2_ind]+"/magnetization_plot.csv"
df2=pd.read_csv(inCsvFile2)
TVec2=np.array(df2["T"])
U_L2=np.array(df2["U_L"])
mask2=(TVec > 1.11) & (TVec<1.15)
TInds2 = np.where(mask2)[0]
T_vec2_binder=TVec2[TInds2]
U_L2_binder=U_L2[TInds2]
print(f"T_vec2_binder={T_vec2_binder}")
print(f"U_L2_binder={U_L2_binder}")
# 2 data points for N0
x1_ind=0
x2_ind=-1
x1=T_vec0_binder[x1_ind]
x2=T_vec0_binder[x2_ind]
print(f"x1={x1}")
print(f"x2={x2}")
y1=U_L0_binder[x1_ind]
y2=U_L0_binder[x2_ind]
print(f"y1={y1}")
print(f"y2={y2}")

# 2 data points for N1
x3_ind=0
x4_ind=-1
x3=T_vec1_binder[x3_ind]
x4=T_vec1_binder[x4_ind]
y3=U_L1_binder[x3_ind]
y4=U_L1_binder[x4_ind]
print(f"x3={x3}")
print(f"x4={x4}")
print(f"y3={y3}")
print(f"y4={y4}")

A0=np.zeros((2,2))
b0=np.array([x1*(y2-y1)/(x2-x1)-y1, x3*(y4-y3)/(x4-x3)-y3])
A0[0,0]=(y2-y1)/(x2-x1)
A0[0,1]=-1

A0[1,0]=(y4-y3)/(x4-x3)

A0[1,1]=-1
solution_0_1=np.linalg.solve(A0,b0)
print(f"solution_0_1={solution_0_1}")

# 2 data points for N2
x5_ind=0
x6_ind=-1
x5=T_vec2_binder[x5_ind]
x6=T_vec2_binder[x6_ind]
print(f"x5={x5}")
print(f"x6={x6}")
y5=U_L2_binder[x5_ind]
y6=U_L2_binder[x6_ind]

print(f"y5={y5}")
print(f"y6={y6}")

A1=np.zeros((2,2))
#
A1[0,0]=(y4-y3)/(x4-x3)

A1[0,1]=-1
A1[1,0]=(y6-y5)/(x6-x5)

A1[1,1]=-1

b1=np.array([x3*(y4-y3)/(x4-x3)-y3,x5*(y6-y5)/(x6-x5)-y5])
solution_1_2=np.linalg.solve(A1,b1)
print(f"solution_1_2={solution_1_2}")