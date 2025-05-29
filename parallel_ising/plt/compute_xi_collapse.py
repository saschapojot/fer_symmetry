import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline, PchipInterpolator
from pathlib import Path
from datetime import datetime
import glob
import sys
import re

import pandas as pd



#this script computes critical exponents using
#data collapse algorithm
#the algorithm is modified based on  paper  A measure of data collapse for scaling
# by Somendra M Bhattacharjee
# J. Phys. A: Math. Gen. 34 (2001) 6375–6380

if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]


Tc=1.13
T_lower=1.14
T_upper=1.16
def load_data_one_N(N):
    """

    :param N:
    :return: rescaled temperature, chi
    """
    csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
    inCsvFile=csvDataFolderRoot+"/magnetization_plot.csv"
    df=pd.read_csv(inCsvFile)
    TVec=np.array(df["T"])
    mask = (TVec > T_lower) & (TVec <T_upper)
    TInds = np.where(mask)[0]

    T_2_collapse=TVec[TInds]
    print(f"N={N}, T_2_collapse={T_2_collapse}")

    chi_each_site_all=np.array(df["chi_each_site"])
    chi_2_collapse=chi_each_site_all[TInds]
    t_2_collapse=np.abs(T_2_collapse-Tc)/Tc
    print(f"chi_2_collapse={chi_2_collapse}")
    return t_2_collapse, chi_2_collapse


NVec=[150,200]
q=2

# N0=NVec[0]
# N1=NVec[1]
# N2=NVec[2]
# t0_vec,chi0_vec=load_data_one_N(N0)
# t1_vec,chi1_vec=load_data_one_N(N1)
# t2_vec,chi2_vec=load_data_one_N(N2)

outDir=f"../dataAll/row{row}/"
Path(outDir).mkdir(exist_ok=True,parents=True)
t_2_collapse_vecs_all=[]
chi_2_collapse_vecs_all=[]

for N in NVec:
    t_2_collapse, chi_2_collapse=load_data_one_N(N)
    t_2_collapse_vecs_all.append(t_2_collapse)
    chi_2_collapse_vecs_all.append(chi_2_collapse)

    spl_func=PchipInterpolator(t_2_collapse,chi_2_collapse)
    fig, ax = plt.subplots()
    plt.scatter(t_2_collapse,chi_2_collapse)
    tVals=np.linspace(t_2_collapse[0],t_2_collapse[-1],50)
    plt.plot(tVals,spl_func(tVals),color="blue")
    plt.xlabel("$t$")
    plt.ylabel("$\chi$")
    plt.title(f"N={N}")
    plt.savefig(outDir+f"/N{N}_chi.png")
    plt.close()

def diff_1_val(ind_NA,ind_NB,d,c):
    """

    :param ind_NA: index for NA,  data from NA by NA lattice are for interpolation
    :param ind_NB: index for NB, data from NB by NB lattice are for comparison
    :param d: trial value of d_exact
    :param c: trial value of c_exact
    :param t_vec: temperature points for interpolation
    :return:
    """

    #A, interpolation
    NA=NVec[ind_NA]
    chi_vec_for_A=chi_2_collapse_vecs_all[ind_NA]

    #collapsed chi for interpolation
    chi_collapsed_vec_for_A=chi_vec_for_A/NA**d

    #collapsed temperature for interpolation
    t_vec_A=t_2_collapse_vecs_all[ind_NA]
    t_vec_A=np.array(t_vec_A)

    t_vec_collapsed_for_A=t_vec_A/NA**c
    func_A=PchipInterpolator(t_vec_collapsed_for_A,chi_collapsed_vec_for_A)

    #B
    chi_vec_for_B_all=chi_2_collapse_vecs_all[ind_NB]
    NB=NVec[ind_NB]
    t_vec_B=t_2_collapse_vecs_all[ind_NB]
    t_vec_B=np.array(t_vec_B)
    t_vec_collapsed_for_B=t_vec_B/NB**c

    t_A_collapse_min=np.min(t_vec_collapsed_for_A)
    t_A_collapse_max=np.max(t_vec_collapsed_for_A)

    #for comparison
    t_B_collapse_overlap_vec=[]
    chi_B_original_overlap_vec=[]
    for ind, val in enumerate(t_vec_collapsed_for_B):
        if val>=t_A_collapse_min and val<=t_A_collapse_max:
            t_B_collapse_overlap_vec.append(val)
            chi_B_original_overlap_vec.append(chi_vec_for_B_all[ind])
    # if len(t_B_collapse_overlap_vec)==0:
    #     print(f"len(t_B_collapse_overlap_vec)={len(t_B_collapse_overlap_vec)}")
    t_B_collapse_overlap_vec_comp=np.array(t_B_collapse_overlap_vec)
    func_A_vals_on_B=func_A(t_B_collapse_overlap_vec_comp)

    chi_B_original_overlap_vec_comp=np.array(chi_B_original_overlap_vec)

    diff_val=np.linalg.norm(func_A_vals_on_B*NB**d-chi_B_original_overlap_vec_comp,ord=q)
    return diff_val**q
d_val_vec=np.linspace(0,5,50)
c_val_vec=np.linspace(-3,0,50)
result_matrix = np.zeros((len(d_val_vec), len(c_val_vec)))
tStart=datetime.now()

for i, d in enumerate(d_val_vec):
    for j, c in enumerate(c_val_vec):
        try:
            for ind0 in range(0,len(NVec)):
                for ind1 in range(ind0+1,len(NVec)):
                    val=diff_1_val(ind0,ind1,d,c)+diff_1_val(ind1,ind0,d,c)
                    result_matrix[i, j]+=val
        except Exception as e:
            print(f"Error at d={d}, c={c}: {e}")
            result_matrix[i, j] = np.nan

# Create heatmap with contour lines
plt.figure(figsize=(10, 8))

# Create the heatmap
plt.pcolormesh(c_val_vec, d_val_vec, result_matrix, cmap='viridis', shading='auto')
colorbar = plt.colorbar(label='Difference value')

# Add contour lines
contour_levels = 100# You can specify exact levels with a list instead
contour = plt.contour(c_val_vec, d_val_vec, result_matrix, levels=contour_levels, colors='white', linewidths=0.8, alpha=0.7)

# Add contour labels (optional)
plt.clabel(contour, inline=True, fontsize=8, fmt='%.3f')

plt.xlabel('c value')
plt.ylabel('d value')
plt.title('Scaling collapse difference for various (c,d) parameters')

# Optionally find the minimum value and mark it
min_idx = np.unravel_index(np.nanargmin(result_matrix), result_matrix.shape)
min_d, min_c = d_val_vec[min_idx[0]], c_val_vec[min_idx[1]]
which_d,which_c=min_idx

min_val = result_matrix[min_idx]
print(f"which_d={which_d}, which_c={which_c}")
print(f"min_d={min_d}, min_c={min_c}")
print(f"min_val={min_val}")

plt.plot(min_c, min_d, 'rx', markersize=10)
plt.annotate(f'Min: ({min_c:.3f}, {min_d:.3f})\nValue: {min_val:.6f}',
             (min_c, min_d), xytext=(10, 10), textcoords='offset points')



plt.tight_layout()
plt.savefig(outDir+"/dc_val.png")


tEnd=datetime.now()
print("time: ",tEnd-tStart)
