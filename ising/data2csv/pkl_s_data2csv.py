import numpy as np
from datetime import datetime
import sys
import re
import glob
import os
import json
from pathlib import Path
import pandas as pd
import pickle


#this script extracts effective data from pkl files
# for s


if (len(sys.argv)!=5):
    print("wrong number of arguments")
    exit()

N=int(sys.argv[1])
TStr=sys.argv[2]
init_path=sys.argv[3]
row=sys.argv[4]
dataRoot=f"./dataAll/N{N}/row{row}/T{TStr}/init_path{init_path}/"

csv_out_path=f"./dataAll/N{N}/row{row}/csvOut_init_path{init_path}/T{TStr}/"


def sort_data_files_by_flushEnd(varName):
    dataFolderName=dataRoot+"/U_s_dataFiles/"+varName+"/"
    dataFilesAll=[]
    flushEndAll=[]
    for oneDataFile in glob.glob(dataFolderName+"/flushEnd*.pkl"):
        dataFilesAll.append(oneDataFile)
        matchEnd=re.search(r"flushEnd(\d+)",oneDataFile)
        if matchEnd:
            flushEndAll.append(int(matchEnd.group(1)))

    endInds=np.argsort(flushEndAll)
    sortedDataFiles=[dataFilesAll[i] for i in endInds]
    return sortedDataFiles


def s_extract_ForOneT(startingFileInd,lag,varName,sweep_to_write):
    TRoot=dataRoot

    sorted_s_DataFilesToRead=sort_data_files_by_flushEnd(varName)
    # print(f"sorted_s_DataFilesToRead={sorted_s_DataFilesToRead}")
    s_StartingFileName=sorted_s_DataFilesToRead[startingFileInd]

    with open(s_StartingFileName,"rb") as fptr:
        s_inArrStart=np.array(pickle.load(fptr))

    s_Arr=s_inArrStart.reshape((sweep_to_write,-1))
    #read the rest of  pkl files

    for pkl_file in sorted_s_DataFilesToRead[(startingFileInd+1):]:
        with open(pkl_file,"rb") as fptr:
            s_inArr=np.array(pickle.load(fptr))
        s_inArr=s_inArr.reshape((sweep_to_write,-1))
        s_Arr=np.concatenate((s_Arr,s_inArr),axis=0)

    s_ArrSelected=s_Arr[::lag,:]

    return s_ArrSelected



def save_s_data(s_ArrSelected,varName):
    outCsvFolder=csv_out_path+"/"
    Path(outCsvFolder).mkdir(exist_ok=True,parents=True)
    outFileName=f"{varName}.csv"
    outCsvFile=outCsvFolder+outFileName
    df=pd.DataFrame(s_ArrSelected)
    # Save to CSV
    print(f"saving {outCsvFile}")
    df.to_csv(outCsvFile, index=False, header=False)


t_save_start=datetime.now()
startingfileIndTmp=50
sweep_to_writeTmp=1000

lagTmp=60
varName="s"
sweep_multiple=3
s_ArrSelected=s_extract_ForOneT(startingfileIndTmp,lagTmp,varName,sweep_to_writeTmp)

save_s_data(s_ArrSelected,varName)


t_save_End=datetime.now()
print(f"time: {t_save_End-t_save_start}")