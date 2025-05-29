import os.path
from pathlib import Path
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
#this script combines M_plot.py for different N

if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]
J_abs=1/2
def load_M_plot_one_N(N,inCsvFile):
    df=pd.read_csv(inCsvFile)
    TVec=np.array(df["T"])
    MValsAll=np.array(df["M"])
    M2ValsAll=np.array(df["M2"])

    return TVec,MValsAll,M2ValsAll


def magnetization_exact(T):
    beta=1/T
    beta_J_2=2*beta*J_abs
    val=np.sinh(beta_J_2)

    return (1-val**(-4))**(1/8)

N_dir_vec=[]
N_vals=[]
data_N_root="../dataAll/"
for NDir in glob.glob(data_N_root+"/N*"):
    matchN=re.search(r"N(\d+)",NDir)
    if matchN:
        N_dir_vec.append(NDir)
        N_vals.append(int(matchN.group(1)))

sortedInds=np.argsort(N_vals)

sorted_N_vals=[N_vals[ind] for ind in sortedInds]

sorted_N_dirs=[N_dir_vec[ind] for ind in sortedInds]
N_out_vec=[]
T_vecs_all=[]
M_vecs_all=[]
M2_vecs_all=[]

for ind in range(0,len(sorted_N_vals)):
    N=sorted_N_vals[ind]
    N_dir=sorted_N_dirs[ind]
    csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"

    inCsvFile=csvDataFolderRoot+"/M_plot.csv"
    exists=os.path.exists(inCsvFile)
    if not exists:
        continue
    TVec,MValsAll,M2ValsAll= load_M_plot_one_N(N,inCsvFile)
    if TVec is None or MValsAll is None or M2ValsAll is None:
        continue
    if N==50 or N==100 or N==150 or N==200 or N==250:
        continue
    N_out_vec.append(N)
    T_vecs_all.append(TVec)
    M_vecs_all.append(MValsAll)
    M2_vecs_all.append(M2ValsAll)

T_mag_exact_vec=np.linspace(1.128,1.134,50)
mag_exact_vec=magnetization_exact(T_mag_exact_vec)
mag2_exact_vec=mag_exact_vec**2
out_dir=f"../dataAll/row{row}/"
Path(out_dir).mkdir(exist_ok=True,parents=True)

plt.figure(figsize=(12, 8))
for ind in range(0,len(T_vecs_all)):
    N=N_out_vec[ind]
    TVec=T_vecs_all[ind]
    MVec=M_vecs_all[ind]
    plt.scatter(TVec,MVec,s=3,label=f"N={N}")
plt.plot(T_mag_exact_vec,mag_exact_vec,color="magenta",linestyle="-",linewidth=0.5)
plt.xlabel("$T$")
plt.ylabel("$|M|$")
plt.legend(loc="best")
plt.title("magnetization")
plt.savefig(out_dir+"/M_compare.png")
plt.close()

plt.figure(figsize=(12, 8))
for ind in range(0,len(T_vecs_all)):
    N=N_out_vec[ind]
    TVec=T_vecs_all[ind]
    M2Vec=M2_vecs_all[ind]
    plt.scatter(TVec,M2Vec,s=3,label=f"N={N}")


plt.plot(T_mag_exact_vec,mag2_exact_vec,color="magenta",linestyle="-",linewidth=0.5)
plt.xlabel("$T$")
plt.ylabel("$M^{2}$")
plt.legend(loc="best")
plt.title("magnetization squared")
plt.savefig(out_dir+"/M2_compare.png")
plt.close()