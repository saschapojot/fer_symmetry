import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
#This script loads M data,
# and plots M for all T


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
mask = (TVec > 0.02)
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