import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
from pathlib import Path
#this script loads auto-correlation of abs magnetization data for each T
# and plots auto-correlation of magnetization for all T


if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()

N=int(sys.argv[1])
init_path=int(sys.argv[2])
row=sys.argv[3]
csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"

TVals=[]
TFileNames=[]

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

auto_corr_abs_M_dir=csvDataFolderRoot+"/corr_abs_M/"
Path(auto_corr_abs_M_dir).mkdir(exist_ok=True,parents=True)
def plt_corr_abs_M_one_T(oneTFile):
    matchT=re.search(r'T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)',oneTFile)
    TStr=matchT.group(1)
    corr_abs_M_csv_file_name=oneTFile+"/abs_M_corr.csv"
    corr_abs_M_arr=np.array(pd.read_csv(corr_abs_M_csv_file_name,header=None))

    plt.figure()
    plt.plot(range(0,len(corr_abs_M_arr)),corr_abs_M_arr,color="blue")
    plt.xlabel("separation")
    plt.ylabel("auto-correlation")
    plt.title(f"auto-correlation of abs M, T={TStr}")
    plt.savefig(auto_corr_abs_M_dir+f"/corr_abs_P_T{TStr}.png")
    plt.close()



tStart=datetime.now()
for k in range(0,len(sortedTFiles)):
    oneTFile=sortedTFiles[k]
    plt_corr_abs_M_one_T(oneTFile)

tEnd=datetime.now()
print(f"time: {tEnd-tStart}")