import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
#this script loads M2, with rescalings
# for all N
if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]
# beta=1/8
nu=1
gamma=1.75
plt.figure()
def plt_one_N(N):
    csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
    inCsvFile=csvDataFolderRoot+"/magnetization_plot.csv"
    df=pd.read_csv(inCsvFile)
    TVec=np.array(df["T"])
    # M2ValsAll=np.array(df["M2"])
    mask = (TVec > 1.16) & (TVec <18)
    Tc=1.12
    TInds = np.where(mask)[0]
    TToPlt=TVec[TInds]
    # M2_to_plot=M2ValsAll[TInds]
    chi_each_site_all=np.array(df["chi_each_site"])
    chi_to_plot=chi_each_site_all[TInds]
    t=np.abs(TToPlt-Tc)/TToPlt
    rescaled_t=N**(1/nu)*t
    # rescaled_m2=M2_to_plot*N**(2*beta/nu)
    rescaled_chi=chi_to_plot*N**(-gamma/nu)
    plt.scatter(rescaled_t,rescaled_chi,label=f"N={N}")
    model=LinearRegression()
    X=np.log(rescaled_t).reshape(-1,1)
    y=np.log(rescaled_chi)
    model.fit(X,y)
    slope=model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)
    print(f"N={N}, slope={slope}, intercept={intercept}, r_squared={r_squared}")


plt.xlabel(r"$tN^{\frac{1}{\nu}}$")
plt.ylabel(r"$\chi N^{-\frac{\gamma}{\nu}}$")
plt.xscale("log")
plt.yscale("log")

NVec=[32,64,128]

for N in NVec:
    plt_one_N(N)
plt.legend(loc="best")
outDir=f"../dataAll/row{row}/"
plt.savefig(outDir+"/rescaled.png")