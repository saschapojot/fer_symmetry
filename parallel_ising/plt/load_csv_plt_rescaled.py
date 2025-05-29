import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats


#this script loads M2, with rescalings

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
# M2ValsAll=np.array(df["M2"])
chi_each_site_all=np.array(df["chi_each_site"])
mask = (TVec > 1.12) & (TVec <1.2)

Tc=1.12
TInds = np.where(mask)[0]
TToPlt=TVec[TInds]
# M2_to_plot=M2ValsAll[TInds]
chi_to_plot=chi_each_site_all[TInds]
t=np.abs(TToPlt-Tc)/TToPlt

# beta=1/8
nu=1
gamma=1
rescaled_t=N**(1/nu)*t

# rescaled_m2=M2_to_plot*N**(2*beta/nu)
rescaled_chi=chi_to_plot*N**(-gamma/nu)
plt.figure()
plt.scatter(rescaled_t,rescaled_chi,color="red")


plt.xlabel(r"$tL^{\frac{1}{\nu}}$")
plt.ylabel(r"$\chi N^{-\frac{\gamma}{\nu}}$")
plt.title(f"N={N}")
plt.savefig(csvDataFolderRoot+"/rescaled.png")
