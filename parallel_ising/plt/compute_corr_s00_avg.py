import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats

#this script computes correlation function
# <s_{00}s_{ij}> for all T
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

def corr_s00_sij_one_T(oneTFile):
    """

    :param oneTFile:
    :return:
    """
    matchT=re.search(r'T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)',oneTFile)
    T_value=float(matchT.group(1))
    s_path=oneTFile+"/s.csv"
    s_arr=np.array(pd.read_csv(s_path,header=None))

    s_multiplied_by_col0=s_arr*(s_arr[:,0][:, np.newaxis])
    s_avg_over_config=np.mean(s_arr,axis=0)

    s00_s_other_corr_avg=np.mean(s_multiplied_by_col0,axis=0)
    s00_covariance=s00_s_other_corr_avg#-s_avg_over_config[0]*s_avg_over_config
    length=len(s00_covariance)
    ind_arr=np.array(range(length))
    i=ind_arr//N
    j=ind_arr%N
    out_arr = np.column_stack((i,j,s00_covariance))
    out_s00_corr_file_name=oneTFile+"/s00_corr.csv"
    df=pd.DataFrame(out_arr)
    df.to_csv(out_s00_corr_file_name, header=False, index=False)

tStart=datetime.now()
for k in range(0,len(sortedTFiles)):
    oneTFile=sortedTFiles[k]
    corr_s00_sij_one_T(oneTFile)
    print(f"{oneTFile}")

tEnd=datetime.now()

print(f"time: {tEnd-tStart}")