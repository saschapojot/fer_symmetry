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
# for M

if (len(sys.argv)!=9):
    print("wrong number of arguments")
    exit()


N=int(sys.argv[1])
TStr=sys.argv[2]
init_path=sys.argv[3]
row=sys.argv[4]
startingfileIndTmp=int(sys.argv[5])
sweep_to_writeTmp=int(sys.argv[6])

lagTmp=int(sys.argv[7])
sweep_multiple=int(sys.argv[8])
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


def M_extract_ForOneT(startingFileInd,lag,varName,sweep_to_write):
    sorted_M_DataFilesToRead=sort_data_files_by_flushEnd(varName)
    M_StaringFileName=sorted_M_DataFilesToRead[startingFileInd]

    with open(M_StaringFileName,"rb") as fptr:
        M_inArrStart=np.array(pickle.load(fptr))

    MVec=M_inArrStart
    for pkl_file in sorted_M_DataFilesToRead[(startingFileInd+1):]:
        with open(pkl_file,"rb") as fptr:
            in_MArr=pickle.load(fptr)
            MVec=np.append(MVec,in_MArr)

    MVecSelected=MVec[::lag]
    return MVecSelected


def save_M_data(MVecSelected,varName):
    outCsvFolder=csv_out_path+"/"
    Path(outCsvFolder).mkdir(exist_ok=True,parents=True)
    outFileName=f"{varName}.csv"
    outCsvFile=outCsvFolder+outFileName

    df=pd.DataFrame(MVecSelected)
    # Save to CSV
    print(f"saving {outCsvFile}")
    df.to_csv(outCsvFile, index=False, header=False)


t_save_start=datetime.now()
varName="M"
MVecSelected=M_extract_ForOneT(startingfileIndTmp,lagTmp,varName,sweep_multiple)

save_M_data(MVecSelected,varName)
t_save_End=datetime.now()
print(f"time: {t_save_End-t_save_start}")