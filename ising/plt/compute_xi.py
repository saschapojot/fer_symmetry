import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
#This script computes xi using second moment method for all T

N=int(sys.argv[1])
init_path=int(sys.argv[2])
row=sys.argv[3]
csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
TVals=[]
TFileNames=[]

unitCellNum=N**2
for TFile in glob.glob(csvDataFolderRoot+"/T*"):

    matchT=re.search(r"T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)",TFile)
    # if float(matchT.group(1))<1:
    #     continue

    if matchT:
        TFileNames.append(TFile)
        TVals.append(float(matchT.group(1)))


sortedInds=np.argsort(TVals)
sortedTVals=[TVals[ind] for ind in sortedInds]
sortedTFiles=[TFileNames[ind] for ind in sortedInds]

def xi_one_T(oneTFile):
    """

    :param oneTFile:
    :return: xi
    """
    # matchT=re.search(r'T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)',oneTFile)
    # T_value=float(matchT.group(1))
    # print(f"T={T_value}")
    s_path=oneTFile+"/s.csv"
    s_arr=np.array(pd.read_csv(s_path,header=None))

    s_multiplied_by_col0=s_arr*(s_arr[:,0][:, np.newaxis])
    s00_s_other_corr_avg=np.mean(s_multiplied_by_col0,axis=0)
    G_arr=s00_s_other_corr_avg.reshape((N,N))

    S_arr=np.fft.ifft2(G_arr)*N**2

    S_arr=np.abs(S_arr)
    S0 = np.sum(G_arr)
    Sk_min = (S_arr[1, 0] + S_arr[0, 1]) / 2  # Averaged |S(k_min)|
    # Compute xi
    k_min = 2 * np.pi / N
    xi = (1 / (2 * np.sin(k_min / 2))) * np.sqrt(S0 / Sk_min - 1)
    return xi



tStart=datetime.now()
# xi=xi_one_T(sortedTFiles[9])
# print(f"xi={xi}")
T_vec=[]
xi_vec=[]
for k in range(0,len(sortedTFiles)):

    oneTFile=sortedTFiles[k]
    matchT=re.search(r'T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)',oneTFile)
    T_value=float(matchT.group(1))
    if T_value<1.12 or T_value>1.19:
        continue

    xi_oneT_val=xi_one_T(oneTFile)
    T_vec.append(T_value)
    xi_vec.append(xi_oneT_val)
    print(f"oneTFile={oneTFile}")
    print(f"xi={xi_oneT_val}")

out_xi_fileName=csvDataFolderRoot+"/xi.csv"

df=pd.DataFrame({"T":T_vec,"xi":xi_vec})
df.to_csv(out_xi_fileName, index=False)


tEnd=datetime.now()

print(f"time: {tEnd-tStart}")