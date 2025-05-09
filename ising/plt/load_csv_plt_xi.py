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



#this script loads xi csv

if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()

N=int(sys.argv[1])
init_path=int(sys.argv[2])
row=sys.argv[3]
csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
inCsvFile=csvDataFolderRoot+"/xi.csv"

df=pd.read_csv(inCsvFile)
Tc=1.11
TVec=np.array(df["T"])

xiVec=np.array(df["xi"])
mask=TVec>1.12
TVec=TVec[mask]
xiVec=xiVec[mask]


X=np.log((TVec-Tc)).reshape(-1,1)
y=np.log(xiVec)
model = LinearRegression()
model.fit(X,y)
# Extract slope (coefficient) and intercept
slope = model.coef_[0]
intercept = model.intercept_
# Calculate R-squared score of the model
r_squared = model.score(X, y)

print(f"Slope: {slope:.3f}")
print(f"Intercept: {intercept:.3f}")
print(f"R-squared: {r_squared:.3f}")