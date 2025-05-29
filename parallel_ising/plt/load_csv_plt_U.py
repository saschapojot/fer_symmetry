import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
from scipy.special import ellipk
#This script loads avg U data, with confidence interval
# and plots U for all T

if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()

N=int(sys.argv[1])
init_path=int(sys.argv[2])
row=sys.argv[3]
csvDataFolderRoot=f"./dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"

inCsvFile=csvDataFolderRoot+"/U_plot.csv"

df=pd.read_csv(inCsvFile)

TVec=np.array(df["T"])
UValsAll=np.array(df["U"])

interval_lowerValsAll=np.array(df["lower"])

interval_upperValsAll=np.array(df["upper"])

U_err_bar=UValsAll-interval_lowerValsAll
# print(f"np.mean(U_err_bar)={np.mean(U_err_bar)}")
mask = (TVec > 0.8) & (TVec <18)
TInds = np.where(mask)[0]
TInds=TInds[::1]
print(f"TInds={TInds}")
TToPlt=TVec[TInds]
print(TToPlt)

J_abs=1/2
def E_per_site_exact(T):
    beta=1/T
    kappa=2*np.sinh(2*beta*J_abs)/(np.cosh(2*beta*J_abs))**2
    K1= ellipk(kappa**2)

    val=-J_abs*1/np.tanh(2*beta*J_abs)\
        *(1+2/np.pi*(2*np.tanh(2*beta*J_abs)**2-1)*K1)

    return val

U_exact_plot=[E_per_site_exact(T) for T in TToPlt]
#plt U
fig,ax=plt.subplots()

ax.errorbar(TToPlt,UValsAll[TInds],
            yerr=U_err_bar[TInds],fmt='o',color="black",
            ecolor='r', capsize=0.1,label='mc',
            markersize=3)
ax.plot(TToPlt,U_exact_plot,color="green",linestyle="--")
print(f"UValsAll[TInds]={UValsAll[TInds]}")
print(f"U_exact_plot={U_exact_plot}")
# ax.scatter(TToPlt,UValsAll[TInds],marker="o",color="black",label='mc',s=1)
# ax.set_xscale("log")
ax.set_xlabel('$T$')
ax.set_ylabel("U")
ax.set_title("U per unit cell, unit cell number="+str(N**2))
plt.legend(loc="best")
# ax.set_xticks(2,3,4,5,6])

# ax.set_xticklabels(["1", "1.5", "2","3", "4","5","6"])
plt.savefig(csvDataFolderRoot+"/UPerUnitCell.png")
plt.close()