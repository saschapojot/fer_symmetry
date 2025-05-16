import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
import pickle
import statsmodels.api as sm
import warnings

#this script computes auto-correlation for abs magnetization
#for all T
# this file deals with data in the original pkl files
if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()


N=int(sys.argv[1])
init_path=int(sys.argv[2])
row=sys.argv[3]
csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
pkl_data_root=f"../dataAll/N{N}/row{row}/"
csv_data_root=pkl_data_root+f"/csvOut_init_path{init_path}/"
TVals=[]
TFileNames=[]
unitCellNum=N**2
pkl_data_root=f"../dataAll/N{N}/row{row}/"
for TFile in glob.glob(pkl_data_root+"/T*"):

    matchT=re.search(r"T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)",TFile)
    # if float(matchT.group(1))<1:
    #     continue

    if matchT:
        TFileNames.append(TFile)
        TVals.append(float(matchT.group(1)))


sortedInds=np.argsort(TVals)
sortedTVals=[TVals[ind] for ind in sortedInds]
sortedTFiles=[TFileNames[ind] for ind in sortedInds]

def sort_data_files_by_flushEnd(pkl_dir):
    dataFilesAll=[]
    flushEndAll=[]
    for oneDataFile in glob.glob(pkl_dir+"/flushEnd*.pkl"):
        dataFilesAll.append(oneDataFile)
        matchEnd=re.search(r"flushEnd(\d+)",oneDataFile)
        if matchEnd:
            flushEndAll.append(int(matchEnd.group(1)))
    endInds=np.argsort(flushEndAll)
    sortedDataFiles=[dataFilesAll[i] for i in endInds]
    return sortedDataFiles


def concatenate_one_s_pkl_files(sorted_s_dataFilesToRead,startingFileInd,sweep_to_write):
    one_s_StartingFileName=sorted_s_dataFilesToRead[startingFileInd]

    with open(one_s_StartingFileName,"rb") as fptr:
        one_s_inArrStart=np.array(pickle.load(fptr))
    s_Arr=one_s_inArrStart.reshape((sweep_to_write,-1))

    #read the rest of  pkl files
    for pkl_file in sorted_s_dataFilesToRead[(startingFileInd+1):]:
        with open(pkl_file,"rb") as fptr:
            s_inArr=np.array(pickle.load(fptr))
        s_inArr=s_inArr.reshape(((sweep_to_write,-1)))
        s_Arr=np.concatenate((s_Arr,s_inArr),axis=0)

    return s_Arr

def s_to_mean(s_Arr):
    print(f"s_Arr.shape={s_Arr.shape}")
    s_all=np.mean(s_Arr,axis=1)
    return s_all

def auto_corrForOneVec(vec):
    """

    :param colVec: a vector of data
    :return: acfOfVecAbs
    """
    same=False
    NLags=int(len(vec))
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
    try:
        acfOfVec=sm.tsa.acf(vec,nlags=NLags)
    except Warning as w:
        same=True
    acfOfVecAbs=np.abs(acfOfVec)
    #the auto-correlation values correspond t0 lengths 0,1,...,NLags-1
    return acfOfVecAbs

def auto_corr_abs_M_one_T(oneTStr,init_path,startingFileInd,sweep_to_write):
    varName_s="s"
    pkl_s_dir=oneTStr+f"/init_path{init_path}/U_s_dataFiles/{varName_s}/"
    sorted_s_pkl_files=sort_data_files_by_flushEnd(pkl_s_dir)

    s_Arr=concatenate_one_s_pkl_files(sorted_s_pkl_files,startingFileInd,sweep_to_write)
    s_mean_vec=s_to_mean(s_Arr)

    abs_M_vec=np.abs(s_mean_vec)

    acfOfVecAbs=auto_corrForOneVec(abs_M_vec)
    matchT=re.search(r'T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)',oneTFile)

    TStr=matchT.group(1)
    csv_out_dir=csv_data_root+f"/T{TStr}/"
    out_corr_file_name=csv_out_dir+"/abs_M_corr.csv"
    df=pd.DataFrame(acfOfVecAbs)
    df.to_csv(out_corr_file_name,header=False, index=False)


sweep_to_writeTmp=100
tStart=datetime.now()
startingfileIndTmp=5
for k in range(0,len(sortedTFiles)):
    oneTFile=sortedTFiles[k]
    auto_corr_abs_M_one_T(oneTFile,init_path,startingfileIndTmp,sweep_to_writeTmp)


tEnd=datetime.now()

print(f"time: {tEnd-tStart}")