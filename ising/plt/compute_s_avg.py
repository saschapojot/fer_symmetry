import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats


#this script converts s csv files to average for all T
# also computes chi for each site
# computes d log value of chi
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
chi_total_all_T=[]
chi_each_site_all_T=[]
F1_val_all_T=[]
U_L_all_T=[]
def magetization_one_T(oneTFile):
    """

    :param oneTFile: corresponds to one temperature
    :return:
    """
    matchT=re.search(r'T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)',oneTFile)
    T_value=float(matchT.group(1))
    s_path=oneTFile+"/s.csv"
    U_path=oneTFile+"/U.csv"

    df_s=np.array(pd.read_csv(s_path,header=None))
    df_U=np.array(pd.read_csv(U_path,header=None))


    s_avg=np.mean(df_s,axis=0)#average over configurations
    out_s_file_name=oneTFile+"/avg_s_combined.csv"
    out_arr=np.array([
        s_avg
    ])
    df=pd.DataFrame(out_arr)
    df.to_csv(out_s_file_name, header=False, index=False)
    magnetization=np.mean(s_avg)

    print(f"magnetization={magnetization}")

    #compute total magnetic dipole, by summing up all s in 1 configuration
    tot_m_vec=np.sum(df_s,axis=1)
    # print(f"tot_m_vec.shape={tot_m_vec.shape}")
    tot_m2_vec=tot_m_vec**2
    E_tot_m2_vec=np.mean(tot_m2_vec)#E(m^{2})
    E_m=np.mean(tot_m_vec)
    # print(f"T_value={T_value}")
    # print(f"E_m={E_m}")
    chi_val_total=(E_tot_m2_vec-E_m**2)/T_value
    chi_val_each_site=(E_tot_m2_vec-E_m**2)/T_value/N**2
    beta=1/T_value
    #compute dbeta chi1
    E_H0=np.mean(df_U)
    E_m2_H0=np.mean(tot_m2_vec*df_U)
    d_beta_chi1=E_tot_m2_vec-beta*E_m2_H0+beta*E_tot_m2_vec*E_H0

    #compute dbeta chi 2
    E_m_H0=np.mean(tot_m_vec*df_U)

    d_beta_chi2=-E_H0**2+2*beta*E_m*E_m_H0\
                -2*beta*E_m**2*E_H0

    d_beta_chi=d_beta_chi1+d_beta_chi2
    dT_chi=-beta**2*d_beta_chi
    F1_val=dT_chi/chi_val_total

    s_avg_over_sites=np.mean(df_s,axis=1)
    s_avg_over_sites_quadratic=s_avg_over_sites**2
    s_avg_over_sites_quartic=s_avg_over_sites**4
    U_L_up=np.mean(s_avg_over_sites_quartic)
    U_L_down=3*(np.mean(s_avg_over_sites_quadratic))**2
    U_L_tmp=1-U_L_up/U_L_down
    magnetization_all_configs=np.mean(df_s,axis=1)
    M_rms=np.sqrt(np.mean(magnetization_all_configs**2))
    return np.abs(magnetization),chi_val_total,chi_val_each_site,F1_val,U_L_tmp,M_rms

tStart=datetime.now()
F1_inv_vals_all_T=[]
M_rms_all_T=[]
for k in range(0,len(sortedTFiles)):
    oneTFile=sortedTFiles[k]
    s_abs,chi_val_total,chi_val_each_site,F1_val,U_L_tmp,M_rms=magetization_one_T(oneTFile)
    magnetization_abs_all.append(s_abs)
    chi_total_all_T.append(chi_val_total)
    chi_each_site_all_T.append(chi_val_each_site)
    F1_val_all_T.append(F1_val)
    F1_inv_vals_all_T.append(1/F1_val)
    U_L_all_T.append(U_L_tmp)
    M_rms_all_T.append(M_rms)



#write magnetization_abs_all
csv_file_name=csvDataFolderRoot+"magnetization_plot.csv"
df=pd.DataFrame({
    "T":sortedTVals,
    "M":magnetization_abs_all,
    "chi_total:":chi_total_all_T,
    "chi_each_site":chi_each_site_all_T,
    "d_T_log_chi":F1_val_all_T,
    "d_T_log_chi_inv":F1_inv_vals_all_T,
    "U_L":U_L_all_T,
    "M_rms":M_rms_all_T
})
df.to_csv(csv_file_name,index=False)

tEnd=datetime.now()

print(f"time: {tEnd-tStart}")