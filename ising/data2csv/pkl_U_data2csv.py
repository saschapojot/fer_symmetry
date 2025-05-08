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
# for U
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

def U_extract_ForOneT(startingFileInd,lag,varName,sweep_to_write):
    sorted_U_DataFilesToRead=sort_data_files_by_flushEnd(varName)

    U_StaringFileName=sorted_U_DataFilesToRead[startingFileInd]
    with open(U_StaringFileName,"rb") as fptr:
        U_inArrStart=np.array(pickle.load(fptr))

    UVec=U_inArrStart
    for pkl_file in sorted_U_DataFilesToRead[(startingFileInd+1):]:
        with open(pkl_file,"rb") as fptr:
            in_UArr=pickle.load(fptr)
            UVec=np.append(UVec,in_UArr)
    UVecSelected=UVec[::lag]
    return UVecSelected


def save_U_data(UVecSelected,varName):
    outCsvFolder=csv_out_path+"/"
    Path(outCsvFolder).mkdir(exist_ok=True,parents=True)
    outFileName=f"{varName}.csv"

    outCsvFile=outCsvFolder+outFileName

    df=pd.DataFrame(UVecSelected)
    # Save to CSV
    print(f"saving {outCsvFile}")
    df.to_csv(outCsvFile, index=False, header=False)


t_save_start=datetime.now()
startingfileIndTmp=50
sweep_multiple=3
lagTmp=60
varName="U"

UVecSelected=U_extract_ForOneT(startingfileIndTmp,lagTmp,varName,sweep_multiple)

save_U_data(UVecSelected,varName)
t_save_End=datetime.now()
print(f"time: {t_save_End-t_save_start}")