import os.path

import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats


#This script loads csv data of M, with confidence interval,
# and computes mean value for each T


if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()

N=int(sys.argv[1])
init_path=int(sys.argv[2])
row=sys.argv[3]
csvDataFolderRoot=f"./dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
TVals=[]
TFileNames=[]
sep=2
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
print(sortedTFiles)
def generate_one_M_point(oneTFile):
    """

    :param oneTFile: corresponds to one temperature
    :return:
    """
    matchT=re.search(r'T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)',oneTFile)
    TVal=float(matchT.group(1))
    M_distPath=oneTFile+"/M.csv"

    df=pd.read_csv(M_distPath,header=None)
    MVec=np.array(df.iloc[:,0])
    MVec=MVec[::sep]
    print("T="+str(TVal)+", data num="+str(len(MVec)))
    M_mean=np.mean(np.abs(MVec))
    M2_mean=np.mean(MVec**2)
    return M_mean,M2_mean


MValsAll=[]
M2ValsAll=[]
tStart=datetime.now()
T_out_vec=[]
for k in range(0,len(sortedTFiles)):
    oneTFile=sortedTFiles[k]
    matchT=re.search(r"T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)",oneTFile)
    T_val=matchT.group(1)
    exists=os.path.exists(oneTFile+"/M.csv")
    if not exists:
        continue
    T_out_vec.append(T_val)
    M_mean,M2_mean=generate_one_M_point(oneTFile)
    MValsAll.append(M_mean)
    M2ValsAll.append(M2_mean)


csv_file_name=csvDataFolderRoot+"M_plot.csv"

df=pd.DataFrame({
    "T":T_out_vec,
    "M":MValsAll,
    "M2":M2ValsAll
})

df.to_csv(csv_file_name,index=False)

tEnd=datetime.now()

print(f"time: {tEnd-tStart}")