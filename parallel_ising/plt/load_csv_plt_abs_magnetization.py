import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
#This script loads M data and chi data, in magnetization_plot.csv
# and plots M, chi for all T


if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()



N=int(sys.argv[1])
init_path=int(sys.argv[2])
row=sys.argv[3]
csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
inCsvFile=csvDataFolderRoot+"/magnetization_plot.csv"

df=pd.read_csv(inCsvFile)

TVec=np.array(df["T"])
MValsAll=np.array(df["M"])
# M2ValsAll=np.array(df["M2"])
chi_valsAll=np.array(df["chi_each_site"])
# U_L=np.array(df["U_L"])
mask = (TVec > 0.2) & (TVec <18)
TInds = np.where(mask)[0]
TInds=TInds[::1]
print(f"TInds={TInds}")
TToPlt=TVec[TInds]
print(TToPlt)

J_abs=1/2
def magnetization_exact(T):
    beta=1/T
    beta_J_2=2*beta*J_abs
    val=np.sinh(beta_J_2)

    return (1-val**(-4))**(1/8)

Tc=1.13
Tc_exact=2*J_abs/np.log(1+np.sqrt(2))
print(f"Tc_exact={Tc_exact}")
#plt M
fig,ax=plt.subplots()
T_mag_exact_vec=np.linspace(TToPlt[0],Tc,50)
print(f"T_mag_exact_vec[-1]={T_mag_exact_vec[-1]}")
mag_exact_vec=magnetization_exact(T_mag_exact_vec)
ax.plot(T_mag_exact_vec,mag_exact_vec,color="magenta",linestyle="-",linewidth=0.5)
ax.errorbar(TToPlt,MValsAll[TInds],fmt='o',color="black",
            ecolor='r', capsize=0.1,label='mc',
            markersize=2)

ax.set_xlabel('$T$')
ax.set_ylabel("$|M|$")
ax.set_title("norm of magnetization, unit cell number="+str(N**2))
plt.legend(loc="best")
plt.savefig(csvDataFolderRoot+"/M_abs.png")
plt.close()

#plt chi
fig,ax=plt.subplots()
ax.errorbar(TToPlt,chi_valsAll[TInds],fmt='o',color="blue",
            ecolor='r', capsize=0.1,label='mc',
            markersize=1)

ax.set_xlabel('$T$')
ax.set_ylabel("$\chi$")
ax.set_title("$\chi$, unit cell number="+str(N**2))
plt.legend(loc="best")
plt.savefig(csvDataFolderRoot+"/chi.png")
plt.close()

#plt Binder ratio
# fig,ax=plt.subplots()
# ax.errorbar(TToPlt,U_L[TInds],fmt='o',color="blue",
#             ecolor='r', capsize=0.1,label='mc',
#             markersize=1)
#
# ax.set_xlabel('$T$')
# ax.set_ylabel("$U_{L}$")
# ax.set_title("Binder ratio, unit cell number="+str(N**2))
# plt.legend(loc="best")
# plt.savefig(csvDataFolderRoot+"/UL.png")
# plt.close()



# fig,ax=plt.subplots()
# ax.errorbar(TToPlt,M2ValsAll[TInds],fmt='o',color="blue",
#             ecolor='r', capsize=0.1,label='mc',
#             markersize=1)


# ax.set_xlabel('$T$')
# ax.set_ylabel("$|M|^{2}$")
# ax.set_title("magnetization squared, unit cell number="+str(N**2))
# plt.legend(loc="best")
# plt.savefig(csvDataFolderRoot+"/M2.png")
# plt.close()