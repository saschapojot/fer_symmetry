import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
from sympy import *
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
#this script loads dT_M and plot
if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()

N=int(sys.argv[1])
init_path=int(sys.argv[2])
row=sys.argv[3]
csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"

inCsvFile=csvDataFolderRoot+"/dT_M_plot.csv"

df=pd.read_csv(inCsvFile)
TVec_numerical=np.array(df["T"])
M_vec_numerical=np.array(df["M"])

Tc_BR=1.13

t_vec=(Tc_BR-TVec_numerical)/Tc_BR


log_M_vec=np.log(M_vec_numerical)

def y_func(t,beta,D,F):
    val=D+beta*np.log(t)+beta*np.log(1+F*t)

    return val


popt,pcov=curve_fit(y_func,t_vec,log_M_vec)
perr = np.sqrt(np.diag(pcov))  # Parameter uncertainties

beta_fit,D_fit,F_fit=popt

print(f"beta_fit={beta_fit} ± {perr[0]:.4f}, "
      f"D_fit={D_fit} ± {perr[1]:.4f}, "
      f"F_fit={F_fit} ± {perr[2]:.4f}"
      )


def M_fit(T,beta_fit,D_fit,F_fit):
    t=(Tc_BR-T)/Tc_BR

    val=D_fit+beta_fit*np.log(t)+beta_fit*np.log(1+F_fit*t)

    return np.exp(val)

T_plt_vec=np.linspace(TVec_numerical[0],TVec_numerical[-1],30)
M_plt_vec=[M_fit(T,beta_fit,D_fit,F_fit) for T in T_plt_vec]

plt.figure()
plt.scatter(TVec_numerical,M_vec_numerical,color="blue",s=3,label="mc")
plt.plot(T_plt_vec,M_plt_vec,color="magenta",linestyle="--",linewidth=1,label="fit")

plt.xlabel("T")
plt.ylabel("M")
plt.title(f"N={N}")
plt.legend(loc="best")
plt.savefig(csvDataFolderRoot+"/M_fit.png")
plt.close()