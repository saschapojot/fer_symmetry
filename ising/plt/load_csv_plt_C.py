import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
#This script loads C data, with confidence interval
# and plots C for all T


if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()

N=int(sys.argv[1])
init_path=int(sys.argv[2])
row=sys.argv[3]
csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
inCsvFile=csvDataFolderRoot+"/C_plot.csv"

df=pd.read_csv(inCsvFile)

TVec=np.array(df["T"])
CValsAll=np.array(df["C"])

interval_lowerValsAll=np.array(df["lower"])

interval_upperValsAll=np.array(df["upper"])

C_err_bar=CValsAll-interval_lowerValsAll

mask = (TVec>0.9)
TInds = np.where(mask)[0]
TInds=TInds[::1]
# print(f"TInds={TInds}")
TToPlt=TVec[TInds]


#plt C
fig,ax=plt.subplots()

# ax.errorbar(TToPlt,CValsAll[TInds],
#             yerr=C_err_bar[TInds],fmt='o',color="blue",
#             ecolor='magenta', capsize=0.1,label='mc',
#             markersize=1)
ax.errorbar(TToPlt,CValsAll[TInds],fmt='o',color="blue",
            ecolor='magenta', capsize=0.1,label='mc',
            markersize=1)
# ax.set_xscale("log")
ax.set_xlabel('$T$')
ax.set_ylabel("C")
ax.set_title("C per unit cell, unit cell number="+str(N**2))
plt.legend(loc="best")
# ax.set_xticks([1, 1.5,2,3, 4,5,6])

# ax.set_xticklabels(["1", "1.5", "2","3", "4","5","6"])
plt.savefig(csvDataFolderRoot+"/CPerUnitCell.png")
plt.close()


