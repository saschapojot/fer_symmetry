import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats

#this file computes dT chi
if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()


N=int(sys.argv[1])
init_path=int(sys.argv[2])
row=sys.argv[3]
csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
TVals=[]
TFileNames=[]
T_lower=1.13
T_upper=1.2
# unitCellNum=N**2

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



def compute_one_T(oneTFile):
    matchT=re.search(r'T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)',oneTFile)
    T_value=float(matchT.group(1))
    beta=1/T_value
    # print(f"T_value={T_value}")
    s_path=oneTFile+"/s.csv"
    U_path=oneTFile+"/U.csv"

    df_s=np.array(pd.read_csv(s_path,header=None))
    df_U=np.array(pd.read_csv(U_path,header=None).iloc[:,0])

    #compute magnetization by averaging all s in 1 configuration
    M_vec=np.mean(df_s,axis=1)
    M_vec_2=M_vec**2
    energy_vec=df_U

    E_H=np.mean(energy_vec)

    M_vec_abs=np.abs(M_vec)
    E_M=np.mean(M_vec_abs)
    #d chi 1 terms
    E_M2=np.mean(M_vec_2)
    E_M2_H=np.mean(M_vec_2*energy_vec)
    dT_chi1=E_M2-beta*E_M2_H+beta*E_M2*E_H

    #d chi2 terms
    E_M_H=np.mean(M_vec_abs*energy_vec)
    dT_chi2=-E_H**2+2*beta*E_M*E_M_H\
            -2*beta*E_M**2*E_H

    dT_chi=dT_chi1+dT_chi2

    chi=(E_M2-E_M**2)*beta

    return chi,dT_chi


chi_each_site_all_T=[]
dT_chi_each_site_all_T=[]

tStart=datetime.now()
for k in range(0,len(sortedTFiles)):
    oneTFile=sortedTFiles[k]
    print(f"oneTFile={oneTFile}")
    chi,dT_chi=compute_one_T(oneTFile)
    chi_each_site_all_T.append(chi)
    dT_chi_each_site_all_T.append(dT_chi)
    print(f"{oneTFile} processed")


csv_file_name=csvDataFolderRoot+"/dT_chi_plot.csv"

df=pd.DataFrame({
    "T":sortedTVals,
    "chi_each_site":chi_each_site_all_T,
    "dT_chi":dT_chi_each_site_all_T
})

df.to_csv(csv_file_name,index=False)

tEnd=datetime.now()

print(f"time: {tEnd-tStart}")