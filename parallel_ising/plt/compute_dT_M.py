import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats


#this script computes
#M, dT M
if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()


N=int(sys.argv[1])
init_path=int(sys.argv[2])
row=sys.argv[3]
csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
TVals=[]
TFileNames=[]

T_lower=0.9
T_upper=1.1
unitCellNum=N**2
for TFile in glob.glob(csvDataFolderRoot+"/T*"):

    matchT=re.search(r"T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)",TFile)
    # if float(matchT.group(1))<1:
    #     continue

    if matchT:

        T=float(matchT.group(1))
        if T>T_lower and T<T_upper:
            TVals.append(T)
            TFileNames.append(TFile)

sortedInds=np.argsort(TVals)
sortedTVals=[TVals[ind] for ind in sortedInds]
sortedTFiles=[TFileNames[ind] for ind in sortedInds]

magnetization_abs_all=[]

chi_each_site_all_T=[]

U_L_all_T=[]

# magnetization_squared_all=[]

dT_M=[]
def magetization_one_T(oneTFile):
    """

   :param oneTFile: corresponds to one temperature
   :return:
   """
    matchT=re.search(r'T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)',oneTFile)
    T_value=float(matchT.group(1))
    # print(f"T_value={T_value}")
    s_path=oneTFile+"/s.csv"
    U_path=oneTFile+"/U.csv"
    df_s=np.array(pd.read_csv(s_path,header=None))
    df_U=np.array(pd.read_csv(U_path,header=None).iloc[:,0])

    # s_avg=np.mean(df_s,axis=0)#average over configurations

    #compute magnetization by averaging all s in 1 configuration
    M_vec=np.mean(df_s,axis=1)
    # print(f"len(M_vec)={len(M_vec)}")
    M_vec_2=M_vec**2
    energy_vec=df_U
    # print(f"len(energy_vec)={len(energy_vec)}")
    M_vec_abs=np.abs(M_vec)
    # print(f"energy_vec={energy_vec}")
    E_MH=np.mean(M_vec_abs*energy_vec)

    E_M=np.mean(M_vec_abs)
    E_H=np.mean(energy_vec)

    dT_M_one_value=(E_MH-E_M*E_H)/T_value**2



    return dT_M_one_value,np.abs(E_M)

tStart=datetime.now()
for k in range(0,len(sortedTFiles)):
    oneTFile=sortedTFiles[k]
    print(f"oneTFile={oneTFile}")
    dT_M_one_value,M=magetization_one_T(oneTFile)
    magnetization_abs_all.append(M)
    # print(f"dT_M_one_value={dT_M_one_value}")

    dT_M.append(dT_M_one_value)
    print(f"{oneTFile} processed")


#write magnetization_abs_all
csv_file_name=csvDataFolderRoot+"dT_M_plot.csv"

df=pd.DataFrame({
    "T":sortedTVals,
    "dT_M":dT_M,
    "M":magnetization_abs_all
})
df.to_csv(csv_file_name,index=False)

tEnd=datetime.now()

print(f"time: {tEnd-tStart}")