import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sympy import *
#this script fits M for T<Tc , T near Tc

if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]

NVec=[50]
T_lower=0.9
T_upper=1.1
Tc=1.13
J=-1/2
# Tc=-2*J/np.log(1+np.sqrt(2))
def load_M_for_one_N(N):
    """

    :param N:
    :return: loads M
    """
    csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
    inCsvFile=csvDataFolderRoot+"/magnetization_plot.csv"
    df=pd.read_csv(inCsvFile)
    TVec=np.array(df["T"])
    mask = (TVec > T_lower) & (TVec < T_upper)

    TInds = np.where(mask)[0]
    T_return=TVec[TInds]

    MVec=np.array(df["M"])
    M_return=MVec[TInds]

    return T_return, M_return

