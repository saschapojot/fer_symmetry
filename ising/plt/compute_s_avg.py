import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats


#this script converts s csv files to average for all Tif (len(sys.argv)!=4):
if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()


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
magnetization_abs_all=[]

def magetization_one_T(oneTFile):
    """

    :param oneTFile: corresponds to one temperature
    :return:
    """
    matchT=re.search(r'T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)',oneTFile)

    s_path=oneTFile+"/s.csv"

    df_s=np.array(pd.read_csv(s_path,header=None))

    s_avg=np.mean(df_s,axis=0)
    out_s_file_name=oneTFile+"/avg_s_combined.csv"
    out_arr=np.array([
        s_avg
    ])
    df=pd.DataFrame(out_arr)
    df.to_csv(out_s_file_name, header=False, index=False)
    magnetization=np.mean(s_avg)
    print(f"magnetization={magnetization}")
    return np.abs(magnetization)

tStart=datetime.now()

for k in range(0,len(sortedTFiles)):
    oneTFile=sortedTFiles[k]
    s_abs=magetization_one_T(oneTFile)
    magnetization_abs_all.append(s_abs)


#write magnetization_abs_all
csv_file_name=csvDataFolderRoot+"magnetization_plot.csv"
df=pd.DataFrame({
    "T":sortedTVals,
    "M":magnetization_abs_all
})
df.to_csv(csv_file_name,index=False)

tEnd=datetime.now()

print(f"time: {tEnd-tStart}")