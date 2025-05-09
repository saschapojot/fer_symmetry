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
chi_valsAll=np.array(df["chi_each_site"])
U_L=np.array(df["U_L"])
mask = (TVec > 0.8) & (TVec <18)
TInds = np.where(mask)[0]
TInds=TInds[::1]
print(f"TInds={TInds}")
TToPlt=TVec[TInds]
print(TToPlt)

#plt M
fig,ax=plt.subplots()

ax.errorbar(TToPlt,MValsAll[TInds],fmt='o',color="black",
            ecolor='r', capsize=0.1,label='mc',
            markersize=1)

ax.set_xlabel('$T$')
ax.set_ylabel("$|P|$")
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
fig,ax=plt.subplots()
ax.errorbar(TToPlt,U_L[TInds],fmt='o',color="blue",
            ecolor='r', capsize=0.1,label='mc',
            markersize=1)

ax.set_xlabel('$T$')
ax.set_ylabel("$U_{L}$")
ax.set_title("Binder ratio, unit cell number="+str(N**2))
plt.legend(loc="best")
plt.savefig(csvDataFolderRoot+"/UL.png")
plt.close()