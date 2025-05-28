import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import warnings
from scipy.stats import ks_2samp
import sys
import os
import math
from pathlib import Path
from datetime import datetime
import glob
from decimal import Decimal, getcontext
#this script concatenates and plots U from pkl files
def format_using_decimal(value, precision=6):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)

T=1.2
N=23
init_path=0
which_row=1
TStr=format_using_decimal(T)
def sort_data_files_by_flushEnd(oneDir):
    dataFilesAll=[]
    flushEndAll=[]
    for oneDataFile in glob.glob(oneDir+"/flushEnd*.pkl"):
        dataFilesAll.append(oneDataFile)
        matchEnd=re.search(r"flushEnd(\d+)",oneDataFile)
        if matchEnd:
            indTmp=int(matchEnd.group(1))
            flushEndAll.append(indTmp)

    endInds=np.argsort(flushEndAll)
    sortedDataFiles=[dataFilesAll[i] for i in endInds]
    return sortedDataFiles

U_pkl_dir=f"./dataAll/N{N}/row{which_row}/T{TStr}/init_path{init_path}/U_s_dataFiles/U/"
print(U_pkl_dir)
flushEnd_vals_all=[]
file_names_all=[]
for file in glob.glob(U_pkl_dir+"/flushEnd*.pkl"):
    match_num=re.search(r"flushEnd(\d+).U",file)
    if match_num:
        file_names_all.append(file)
        flushEnd_vals_all.append(int(match_num.group(1)))


sortedInds=np.argsort(flushEnd_vals_all)
# print(file_names_all)
sorted_flushEnd_vals_all=[flushEnd_vals_all[ind] for ind in sortedInds]

sorted_file_names_all=[file_names_all[ind] for ind in sortedInds]

startingFileInd=0
startingFileName=sorted_file_names_all[startingFileInd]
# print(sorted_file_names_all)
with open(startingFileName,"rb") as fptr:
    inArrStart=pickle.load(fptr)

U_arr=inArrStart
for pkl_file in sorted_file_names_all[(startingFileInd+1):]:
    with open(pkl_file,"rb") as fptr:
        inArr=pickle.load(fptr)
    U_arr=np.append(U_arr,inArr)


print(f"len(U_arr)={len(U_arr)}")
U_starting_ind=0
print(U_arr[-10:])
plt.figure()
plt.plot(range(U_starting_ind,len(U_arr)),U_arr[U_starting_ind:],color="black")
plt.title(f"U_T{TStr}.png")
plt.savefig(f"U_T{TStr}.png")
plt.close()