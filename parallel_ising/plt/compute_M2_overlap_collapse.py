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
from scipy.interpolate import CubicSpline, PchipInterpolator
from pathlib import Path
#this script loads M2, using data collapse
# for all N


if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]

T_lower=1.04
T_upper=1.13

# plt.figure()
def load_data_one_N(N):
    csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
    inCsvFile=csvDataFolderRoot+"/magnetization_plot.csv"
    df=pd.read_csv(inCsvFile)
    TVec=np.array(df["T"])
    MValsAll=np.array(df["M"])
    M2ValsAll=MValsAll**2

    mask = (TVec > T_lower) & (TVec < T_upper)
    TInds = np.where(mask)[0]
    TToPlt=TVec[TInds]
    M2_to_plot=M2ValsAll[TInds]

    # plt.scatter(TToPlt,M2_to_plot,label=f"N={N}",s=2)
    return TToPlt[::-1],M2_to_plot[::-1]


NVec=[100,150,200]
q=2
outDir=f"../dataAll/row{row}/"
Path(outDir).mkdir(exist_ok=True,parents=True)
Tc=1.14
t_2_collapse_vecs_all=[]
M2_2_collapse_vecs_all=[]
for N in NVec:
    TToPlt,M2_to_plot=load_data_one_N(N)
    t_vec=np.abs(TToPlt-Tc)/Tc

    t_2_collapse_vecs_all.append(t_vec)
    M2_2_collapse_vecs_all.append(M2_to_plot)

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
    M2_vec_for_A=M2_2_collapse_vecs_all[ind_NA]

    M2_collapsed_vec_for_A=M2_vec_for_A*NA**d

    t_vec_A=t_2_collapse_vecs_all[ind_NA]
    t_vec_A=np.array(t_vec_A)

    t_vec_collapsed_for_A=t_vec_A*NA**c
    # print(t_vec_collapsed_for_A)

    #B
    NB=NVec[ind_NB]
    M2_vec_for_B=M2_2_collapse_vecs_all[ind_NB]

    # M2_collapsed_vec_for_B=M2_vec_for_B*NB**d

    t_vec_B=t_2_collapse_vecs_all[ind_NB]
    t_vec_B=np.array(t_vec_B)

    t_vec_collapsed_for_B=t_vec_B*NB**c

    t_A_collapse_min=np.min(t_vec_collapsed_for_A)
    t_A_collapse_max=np.max(t_vec_collapsed_for_A)
    t_B_collapse_min=np.min(t_vec_collapsed_for_B)
    t_B_collapse_max=np.max(t_vec_collapsed_for_B)

    #for interpolation
    t_A_collapse_overlap_vec=[]
    M2_A_collapse_overlap_vec=[]
    for ind,val in enumerate(t_vec_collapsed_for_A):
        if val>=t_B_collapse_min and val<=t_B_collapse_max:
            t_A_collapse_overlap_vec.append(val)
            M2_A_collapse_overlap_vec.append(M2_collapsed_vec_for_A[ind])
    # print(M2_A_collapse_overlap_vec)
    t_A_overlap_vec_interp=np.array(t_A_collapse_overlap_vec)
    M2_A_collapse_overlap_vec_interp=np.array(M2_A_collapse_overlap_vec)
    # print(f"t_A_overlap_vec_interp={t_A_overlap_vec_interp}")
    func_A=PchipInterpolator(t_A_overlap_vec_interp,M2_A_collapse_overlap_vec_interp)

    #for comparison
    t_B_collapse_overlap_vec=[]
    M2_B_original_overlap_vec=[]

    for ind, val in enumerate(t_vec_collapsed_for_B):
        if val>=t_A_collapse_min and val<=t_A_collapse_max:
            t_B_collapse_overlap_vec.append(val)
            M2_B_original_overlap_vec.append(M2_vec_for_B[ind])

    t_B_collapse_overlap_vec_comp=np.array(t_B_collapse_overlap_vec)
    func_A_vals_on_B=func_A(t_B_collapse_overlap_vec_comp)

    M2_B_original_overlap_vec_comp=np.array(M2_B_original_overlap_vec)

    diff_val=np.linalg.norm(func_A_vals_on_B/NB**d-M2_B_original_overlap_vec_comp,ord=1)
    return diff_val**q


ind0=1
ind1=2
nm=diff_1_val(ind0,ind1,0.25,1)
print(f"nm={nm}")


# d_val_vec=np.linspace(0.01,2,30)
# c_val_vec=np.linspace(0.01,2,30)
# result_matrix = np.zeros((len(d_val_vec), len(c_val_vec)))
# tStart=datetime.now()
# for i, d in enumerate(d_val_vec):
#     for j, c in enumerate(c_val_vec):
#         try:
#             for ind0 in range(0,len(NVec)):
#                 for ind1 in range(ind0+1,len(NVec)):
#                     val=diff_1_val(ind1,ind0,d,c)
#                     result_matrix[i, j] =val
#         except Exception as e:
#             print(f"Error at d={d}, c={c}: {e}")
#             result_matrix[i, j] = np.nan
#
#
# # Create heatmap with contour lines
# plt.figure(figsize=(10, 8))
#
# # Create the heatmap
# plt.pcolormesh(c_val_vec, d_val_vec, result_matrix, cmap='viridis', shading='auto')
# colorbar = plt.colorbar(label='Difference value')
#
# # Add contour lines
# contour_levels = 100# You can specify exact levels with a list instead
# contour = plt.contour(c_val_vec, d_val_vec, result_matrix, levels=contour_levels, colors='white', linewidths=0.8, alpha=0.7)
# # Add contour labels (optional)
# plt.clabel(contour, inline=True, fontsize=8, fmt='%.3f')
#
# plt.xlabel('c value')
# plt.ylabel('d value')
# plt.title('Scaling collapse difference for various (c,d) parameters')
#
# # Optionally find the minimum value and mark it
# min_idx = np.unravel_index(np.nanargmin(result_matrix), result_matrix.shape)
# min_d, min_c = d_val_vec[min_idx[0]], c_val_vec[min_idx[1]]
# which_d,which_c=min_idx
#
# min_val = result_matrix[min_idx]
# print(f"which_d={which_d}, which_c={which_c}")
# print(f"min_d={min_d}, min_c={min_c}")
# print(f"min_val={min_val}")
#
# plt.plot(min_c, min_d, 'rx', markersize=10)
# plt.annotate(f'Min: ({min_c:.3f}, {min_d:.3f})\nValue: {min_val:.6f}',
#              (min_c, min_d), xytext=(10, 10), textcoords='offset points')
#
#
#
# plt.tight_layout()
# plt.savefig(outDir+"/dc_val.png")
#
#
# tEnd=datetime.now()
# print("time: ",tEnd-tStart)
#
