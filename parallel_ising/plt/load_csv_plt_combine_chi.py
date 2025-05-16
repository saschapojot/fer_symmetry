import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression


#this script loads chi values for different N
# and make a combined plot
if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]

def load_data_one_N(N):
    """

    :param N:
    :return: temperature, chi
    """
    csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
    inCsvFile=csvDataFolderRoot+"/magnetization_plot.csv"
    df=pd.read_csv(inCsvFile)
    TVec=np.array(df["T"])
    mask = (TVec > 0.8)
    TInds = np.where(mask)[0]
    TToPlt=TVec[TInds]
    chi_each_site_all=np.array(df["chi_each_site"])
    chi_to_plot=chi_each_site_all[TInds]
    return TToPlt,chi_to_plot


NVec=[32,64,128]
N0=NVec[0]
N1=NVec[1]
N2=NVec[2]


T0_vec,chi0_vec=load_data_one_N(N0)
T1_vec,chi1_vec=load_data_one_N(N1)
T2_vec,chi2_vec=load_data_one_N(N2)

plt.figure()
plt.scatter(T0_vec,chi0_vec, color="red",label=f"N={N0}")
plt.scatter(T1_vec,chi1_vec, color="blue",label=f"N={N1}")
plt.scatter(T2_vec,chi2_vec, color="black",label=f"N={N2}")

plt.title("$\chi$ for different N")
plt.xlabel("$T$")
plt.ylabel("$\chi$")
plt.legend(loc="best")
outDir=f"../dataAll/row{row}/"
plt.savefig(outDir+"/chi_all.png")


chi0_max=np.max(chi0_vec)
chi1_max=np.max(chi1_vec)
chi2_max=np.max(chi2_vec)

chi_max=[chi0_max,chi1_max,chi2_max]

X=np.log(NVec).reshape(-1,1)
y=np.log(chi_max)
model=LinearRegression()
model.fit(X,y)
slope=model.coef_[0]
intercept = model.intercept_
r_squared = model.score(X, y)
print(f"slope={slope}, intercept={intercept}, r_squared={r_squared}")