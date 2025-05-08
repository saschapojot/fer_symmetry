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
from pathlib import Path
from scipy.optimize import curve_fit
#this script fits singularity of chi


if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]
N_vec=np.array([32,64,128])

csvDataFolderRoot_vec=[f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/" for N in N_vec]
csv_out_dir=f"../dataAll/row{row}/"
Path(csv_out_dir).mkdir(exist_ok=True,parents=True)

Tc=11.31722889
def chi_func(T,alpha,beta,gamma):
    return alpha*(T-Tc)**(-gamma)+beta

#N0_ind
N0_ind=0#32
#N0_ind
inCsvFile0=csvDataFolderRoot_vec[N0_ind]+"/magnetization_plot.csv"
df0=pd.read_csv(inCsvFile0)
TVec0=np.array(df0["T"])
chi0=np.array(df0["chi_each_site"])
mask0=(TVec0>11.38) &( TVec0<12)
TInds0 = np.where(mask0)[0]
T_vec0_fit=TVec0[TInds0]
chi0_fit=chi0[TInds0]
print(f"T_vec0_fit={T_vec0_fit}")
print(f"chi0_fit={chi0_fit}")
# init_guess0=[1,1,-1]
# # Perform the curve fit. for N0
# popt0, pcov0 = curve_fit(chi_func, T_vec0_fit, chi0_fit, p0=init_guess0,maxfev=10000)
# # Extract the best-fit parameters.
# alpha_fit0, beta_fit0, gamma_fit0 = popt0
# print(f"Fitted parameters for {N_vec[N0_ind]}:\n alpha0 = {alpha_fit0}\n beta0  = {beta_fit0}\n gamma0 = {gamma_fit0}")



#N1_ind
N1_ind=1#64
inCsvFile1=csvDataFolderRoot_vec[N1_ind]+"/magnetization_plot.csv"
df1=pd.read_csv(inCsvFile1)
TVec1=np.array(df1["T"])
chi1=np.array(df1["chi_each_site"])
mask1=(TVec1>11.33)
TInds1 = np.where(mask1)[0]
T_vec1_fit=TVec1[TInds1]
chi1_fit=chi1[TInds1]
print(f"T_vec1_fit={T_vec1_fit}")
print(f"chi1_fit={chi1_fit}")
init_guess1=[1,1,-1]
# popt1, pcov1 = curve_fit(chi_func, T_vec1_fit, chi1_fit, p0=init_guess1,maxfev=10000)
# # Extract the best-fit parameters.
# alpha_fit1, beta_fit1, gamma_fit1 = popt1
# print(f"Fitted parameters for {N_vec[N1_ind]}:\n alpha1 = {alpha_fit1}\n beta1  = {beta_fit1}\n gamma1 = {gamma_fit1}")

#N2_ind
N2_ind=2#128
inCsvFile2=csvDataFolderRoot_vec[N2_ind]+"/magnetization_plot.csv"
df2=pd.read_csv(inCsvFile2)
TVec2=np.array(df2["T"])
chi2=np.array(df2["chi_each_site"])
mask2=(TVec2>11.33)
TInds2 = np.where(mask2)[0]
T_vec2_fit=TVec2[TInds2]
chi2_fit=chi2[TInds2]
print(f"T_vec2_fit={T_vec2_fit}")
print(f"chi2_fit={chi2_fit}")


init_guess2=[1,1,-1]
# popt2, pcov2 = curve_fit(chi_func, T_vec2_fit, chi2_fit, p0=init_guess2,maxfev=10000)
# # Extract the best-fit parameters.
# alpha_fit2, beta_fit2, gamma_fit2 = popt2
# print(f"Fitted parameters for {N_vec[N2_ind]}:\n alpha2 = {alpha_fit2}\n beta2  = {beta_fit2}\n gamma2 = {gamma_fit2}")

#for N1
beta_estimate1=0
T1_fit_linear=np.log(T_vec1_fit-Tc)
model1=LinearRegression()
model1.fit(T1_fit_linear.reshape(-1,1),np.log(chi1_fit-beta_estimate1))
# Extract the slope and intercept
slope1 = model1.coef_[0]
intercept1 = model1.intercept_
print(f"Slope1: {slope1:.3f}, Intercept1: {intercept1:.3f}")

#for N2
beta_estimate2=0
T2_fit_linear=np.log(T_vec2_fit-Tc)
model2=LinearRegression()
model2.fit(T2_fit_linear.reshape(-1,1),np.log(chi2_fit-beta_estimate2))
# Extract the slope and intercept
slope2 = model2.coef_[0]
intercept2 = model2.intercept_
print(f"Slope2: {slope2:.3f}, Intercept2: {intercept2:.3f}")



chi0_max=38.0525885224236
chi1_max=134.352103730896
chi2_max=268.708344191767
chi_max_vec=[chi0_max,chi1_max,chi2_max]
model_L=LinearRegression()
model_L.fit(np.log(np.array(N_vec)).reshape(-1,1),np.log(chi_max_vec))
slopeL = model_L.coef_[0]
interceptL = model_L.intercept_
print(f"SlopeL: {slopeL:.3f}, InterceptL: {interceptL:.3f}")



# M_rms0=0.709984206183413
# M_rms1=0.664614534592322
# M_rms2=0.650996959954183
# M_rms_vec=np.array([M_rms0,M_rms1,M_rms2])
# model_M=LinearRegression()
# model_M.fit(np.log(N_vec).reshape(-1,1),np.log(M_rms_vec))
# slopeM = model_M.coef_[0]
# interceptM = model_M.intercept_
# print(f"SlopeM: {slopeM:.3f}, InterceptM: {interceptM:.3f}")