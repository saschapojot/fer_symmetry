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
#this script combines M_plot.py for one N
# regression of W over 1/log b


N_base_vec=[5,6,7,8]

b_multiple_vec=[2,3,4,5,6,7,8]


if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]

def load_M_plot_one_N(N,inCsvFile):
    df=pd.read_csv(inCsvFile)
    TVec=np.array(df["T"])
    MValsAll=np.array(df["M"])
    M2ValsAll=np.array(df["M2"])

    return TVec,MValsAll,M2ValsAll


TStr="1.134"


Nbase=8
#read NBase M_plot.py
csvDataFolderRoot_base=f"../dataAll/N{Nbase}/row{row}/csvOut_init_path{init_path}/"
inCsvFile_base=csvDataFolderRoot_base+"/M_plot.csv"
df_base=pd.read_csv(inCsvFile_base)
selected_row_base = df_base[df_base['T'] == float(TStr)]

M2_base=float(selected_row_base["M2"])
print(M2_base)

M2_b_multiply_vec=[]
for b in b_multiple_vec:
    bN=b*Nbase
    csvDataFolderRoot=f"../dataAll/N{bN}/row{row}/csvOut_init_path{init_path}/"
    inCsvFile=csvDataFolderRoot+"/M_plot.csv"
    exists=os.path.exists(inCsvFile)
    if not exists:
        continue
    df=pd.read_csv(inCsvFile)
    selected_row = df[df['T'] == float(TStr)]
    # print(f"bN={bN}")
    # print(selected_row["M2"])
    M2_tmp=float(selected_row["M2"])
    M2_b_multiply_vec.append(M2_tmp)

print(M2_b_multiply_vec)
#M2_b_multiply_vec  is for b=2,3,..., 8
W_vec=[]
for ind in range(0,len(b_multiple_vec)):
    b=b_multiple_vec[ind]
    M2_b_mul_tmp=M2_b_multiply_vec[ind]
    W2_val=np.log(M2_b_mul_tmp/M2_base)/np.log(b)
    W_vec.append(W2_val)

inv_log_b_vec=[1/np.log(b) for b in b_multiple_vec]

out_dir=f"../dataAll/row{row}/Nbase{Nbase}/"
Path(out_dir).mkdir(exist_ok=True,parents=True)
plt.figure()
plt.scatter(inv_log_b_vec,W_vec,color="black",label=f"N={Nbase}")
plt.xlabel("$1/\log(b)$")
plt.ylabel("$W$")
plt.legend(loc="best")
plt.savefig(out_dir+f"/W_N{Nbase}_T{TStr}.png")
plt.close()