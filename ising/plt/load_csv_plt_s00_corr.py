import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
#this script loads correlation function csv files
# for s00 and s_{ij}
# for all T


if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()

N=int(sys.argv[1])
init_path=int(sys.argv[2])
row=sys.argv[3]
csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
inCsvFile=csvDataFolderRoot+"/magnetization_plot.csv"
s00_corr_dir=csvDataFolderRoot+"/s00_corr/"
s00_fit_dir=csvDataFolderRoot+"/s00_fit/"
Path(s00_corr_dir).mkdir(exist_ok=True,parents=True)
Path(s00_fit_dir).mkdir(exist_ok=True,parents=True)

TVals=[]
TFileNames=[]
for TFile in glob.glob(csvDataFolderRoot+"/T*"):

    matchT=re.search(r"T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)",TFile)
    # if float(matchT.group(1))<1:
    #     continue

    if matchT:
        TFileNames.append(TFile)
        TVals.append(float(matchT.group(1)))

sortedInds=np.argsort(TVals)
sortedTVals=[TVals[ind] for ind in sortedInds]
sortedTFiles=[TFileNames[ind] for ind in sortedInds]


def plt_s00_corr_one_T(oneTFile):
    matchT=re.search(r'T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)',oneTFile)
    TStr=matchT.group(1)
    if float(TStr)<=0.8:
        return
    s00_corr_csv_file_name=oneTFile+"/s00_corr.csv"

    s00_corr_arr=np.array(pd.read_csv(s00_corr_csv_file_name,header=None))

    dx=np.minimum(s00_corr_arr[:,0],N-s00_corr_arr[:,0])
    dy=np.minimum(s00_corr_arr[:,1],N-s00_corr_arr[:,1])

    distances=np.sqrt(dx**2+dy**2)
    unique_distances = np.unique(distances)
    # print(len(unique_distances))
    # Compute the average s00_corr for each unique distance
    avg_corr = np.array([
        np.mean(s00_corr_arr[distances == d, 2]) for d in unique_distances
    ])
    ###fit distances vs s00_corr_arr
    # mask=(distances>10) & (distances<20)
    # filtered_distances = distances[mask]
    # filtered_corr = s00_corr_arr[mask, 2]
    # # Ensure no non-positive values (logarithm is not defined for zero or negative values)
    # valid = (filtered_distances > 0) & (filtered_corr > 0)
    # filtered_distances = filtered_distances[valid]
    # filtered_corr = filtered_corr[valid]
    # # Take the logarithm of the filtered data
    # log_dist = np.log(filtered_distances)
    # log_corr = np.log(filtered_corr)
    # X = log_dist.reshape(-1, 1)
    # y = log_corr
    # # Create and fit the linear regression model
    # model = LinearRegression()
    # model.fit(X, y)

    ###fit unique_distances vs avg_corr
    mask=(unique_distances>10) &(unique_distances<np.sqrt(2)*N/1.5)
    filtered_unique_distances=unique_distances[mask]
    filtered_avg_corr=avg_corr[mask]
    valid=(filtered_unique_distances>0) &(filtered_avg_corr>0)
    filtered_unique_distances=filtered_unique_distances[valid]
    filtered_avg_corr=filtered_avg_corr[valid]
    log_unique_dist=np.log(filtered_unique_distances)
    log_avg_corr=np.log(filtered_avg_corr)
    X=log_unique_dist.reshape(-1,1)
    y=log_avg_corr
    model = LinearRegression()
    model.fit(X, y)
    # Extract slope (coefficient) and intercept
    slope = model.coef_[0]
    intercept = model.intercept_
    # Calculate R-squared score of the model
    r_squared = model.score(X, y)

    # Print regression statistics
    print("======================")
    print(f"T={TStr}")
    print(f"Slope: {slope:.3f}")
    print(f"Intercept: {intercept:.3f}")
    print(f"R-squared: {r_squared:.3f}")



    plt.scatter(unique_distances, avg_corr, color='red', alpha=0.8)
    plt.xlabel('Distance from [0,0]')
    plt.ylabel('$<s_{00}s_{ij}>$')
    plt.title(r'Scatter Plot of Distance vs $<s_{00}s_{ij}>$ at T = '+str(TStr))
    plt.grid(True)
    plt.savefig(s00_corr_dir+f"/s00_corr_T{TStr}.png")
    plt.close()

    #plot log log fit
    if r_squared>0.1:
        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, color='blue', alpha=0.7)
        # Generate points for the regression line
        x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = model.predict(x_line)
        plt.plot(x_line, y_line, label=f'Fit: slope = {slope:.3f}', color='red')
        plt.xlabel('log(Distance)')
        plt.ylabel('log(s00_corr)')
        plt.title(f'Log-Log Linear Regression at T = {TStr}')
        plt.legend()
        plt.grid(True)
        plt.savefig(s00_fit_dir+f"/s00_fit_T{TStr}.png")
        plt.close()




tStart=datetime.now()
# print(sortedTFiles[0])
# plt_s00_corr_one_T(sortedTFiles[0])

for k in range(0,len(sortedTFiles)):
    oneTFile=sortedTFiles[k]
    plt_s00_corr_one_T(oneTFile)
    print(f"{oneTFile}")

tEnd=datetime.now()

print(f"time: {tEnd-tStart}")