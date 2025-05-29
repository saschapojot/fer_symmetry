import os.path
from pathlib import Path
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
#this script combines M_plot.py for all different N
# regression of W over 1/log b


N_base_vec=[5,6,7]

b_multiple_vec=[2,3,4,5,6]
TStr_vec=["1.129","1.13","1.132","1.133","1.134"]
if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]
out_dir=f"../dataAll/row{row}/"
Path(out_dir).mkdir(exist_ok=True,parents=True)
for TStr in TStr_vec:
    #for each TStr make one combined plot
    plt.figure(figsize=(12,12))
    for Nbase in N_base_vec:
        #read NBase M_plot.py
        csvDataFolderRoot_base=f"../dataAll/N{Nbase}/row{row}/csvOut_init_path{init_path}/"
        inCsvFile_base=csvDataFolderRoot_base+"/M_plot.csv"
        df_base=pd.read_csv(inCsvFile_base)
        selected_row_base = df_base[df_base['T'] == float(TStr)]
        # print(f"Nbase={Nbase}")
        # print(selected_row_base)
        M2_base=float(selected_row_base["M2"])

        #b multiples
        M2_b_multiply_vec=[]
        for b in b_multiple_vec:
            bN=b*Nbase
            csvDataFolderRoot=f"../dataAll/N{bN}/row{row}/csvOut_init_path{init_path}/"
            inCsvFile=csvDataFolderRoot+"/M_plot.csv"
            exists=os.path.exists(inCsvFile)
            if not exists:
                continue
            df=pd.read_csv(inCsvFile)
            selected_row = df[df['T'] == float(TStr)]
            M2_tmp=float(selected_row["M2"])
            M2_b_multiply_vec.append(M2_tmp)

        #M2_b_multiply_vec  is for b=2,3,..., 8
        W_vec=[]
        for ind in range(0,len(b_multiple_vec)):
            b=b_multiple_vec[ind]
            M2_b_mul_tmp=M2_b_multiply_vec[ind]
            W2_val=np.log(M2_b_mul_tmp/M2_base)/np.log(b)
            W_vec.append(W2_val)
        inv_log_b_vec=[1/np.log(b) for b in b_multiple_vec]
        plt.scatter(inv_log_b_vec,W_vec,label=f"N={Nbase}")

        #fit
        # print(W_vec)
        last_num=5
        first_num=3
        W_vec_2_fit=np.array(W_vec[-last_num:])
        inv_log_b_vec_2_fit=np.array(inv_log_b_vec[-last_num:])
        X=inv_log_b_vec_2_fit.reshape(-1,1)
        y=W_vec_2_fit
        model = LinearRegression()
        # Fit the model to your data
        model.fit(X, y)
        # Access model coefficients and intercept
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)
        print("R² Score:", model.score(X, y))
        X_plt=np.linspace(0,1,40)
        X_plt_pred=X_plt.reshape(-1,1)
        y_plt=model.predict(X_plt_pred)

        plt.plot(X_plt,y_plt)





    plt.xlabel("$1/\log(b)$")
    plt.ylabel("$W$")
    plt.legend(loc="best")
    plt.title(f"T={TStr}")
    plt.savefig(out_dir+f"/W_T{TStr}_all_Nbase.png")
    plt.close()
