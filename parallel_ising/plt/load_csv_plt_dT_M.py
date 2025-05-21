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


J=-Rational(1,2)

T=symbols("T",cls=Symbol,real=True)

beta=1/T

M_exact=(1-(sinh(2*beta*J)**(-4)))**(Rational(1,8))

dT_M_exact=diff(M_exact,T)
dlog_M_exact=dT_M_exact/M_exact

func_dT_M=lambdify(T,dT_M_exact,"numpy")
func_M=lambdify(T,M_exact,"numpy")
func_dlog_M=lambdify(T,dlog_M_exact,"numpy")

df=pd.read_csv(inCsvFile)
TVec_numerical=np.array(df["T"])
dT_M_vec_numerical=np.array(df["dT_M"])
M_vec_numerical=np.array(df["M"])
# print(f"M_vec_numerical={M_vec_numerical}")
T_vec_for_exact=np.linspace(TVec_numerical[0],TVec_numerical[-1],15)
dlog_M_exact=func_dlog_M(T_vec_for_exact)
dT_M_vec_exact=func_dT_M(T_vec_for_exact)

plt.figure()
plt.scatter(TVec_numerical,dT_M_vec_numerical,color="black",s=2,label="MC")
plt.plot(T_vec_for_exact,dT_M_vec_exact,color="magenta", linestyle="--", linewidth=1,label="exact")

plt.xlabel("$T$")
plt.ylabel("$\partial_{T}M$")
plt.title(f"N={N}")

plt.legend(loc="best")

plt.savefig(csvDataFolderRoot+f"/dt_M_N{N}.png")
plt.close()


# T_vals=TVec
# T_vals=T_vec_for_exact
# n=len(TVec)-1
# print(f"n={n}")
F1_vals=dT_M_vec_numerical/M_vec_numerical
# F1_vals=dT_M_vec_exact/M_vec_exact
plt.figure()
plt.scatter(TVec_numerical,F1_vals,color="blue",s=2,label="MC")
plt.plot(T_vec_for_exact,dlog_M_exact,color="red",linestyle="--", linewidth=1,label="exact")
plt.xlabel("$T$")
plt.ylabel("$\partial_{T}M/M$")
plt.title(f"N={N}")
plt.legend(loc="best")
plt.savefig(csvDataFolderRoot+f"/dlog_M_N{N}.png")
plt.close()
Tc_BR=1.13

def y_func_rational_N3(T,beta,b2,b3):
    val_up=-1-2*b2*(Tc_BR-T)-3*b3*(Tc_BR-T)**2

    val_down=Tc_BR-T+b2*(Tc_BR-T)**2+b3*(Tc_BR-T)**3

    return beta*val_up/val_down

popt_N3,pcov_N3=curve_fit(y_func_rational_N3,TVec_numerical,F1_vals)
perr_N3 = np.sqrt(np.diag(pcov_N3))  # Parameter uncertainties
beta_fit_N3,b2_fit_N3,b3_fit_N3=popt_N3
print(f"beta_fit_N3={beta_fit_N3} ± {perr_N3[0]:.4f}, b2_fit_N3={b2_fit_N3} ± {perr_N3[1]:.4f}, b3_fit_N3={b3_fit_N3} ± {perr_N3[2]:.4f}")

#TT algorithm part
#
# B=[]
# m0=0
# T_ind=[m0]
# S=[j for j in range(1,n+1)]
#
# def t0(n):
#     """
#
#     :param n:
#     :return: F1_vals[n]-F1_vals[0]
#     """
#     return  F1_vals[n]-F1_vals[0]
#
#
# t0_vec=[t0(n) for n in range(0,len(T_vals))]
# #j=1
# m1_ind=0#index of m1 in S
# m1=S[m1_ind]
# # print(m1)
#
# while np.abs(t0_vec[m1])==0 and m1_ind<len(S):
#     m1_ind+=1
#     m1=S[m1_ind]
# # print(S)
# S.pop(m1_ind)
# # print(S)
# T_ind.append(m1)
# # print(T_ind)
# x1_p=T_vals[m1]
# x0_p=T_vals[m0]
# t_minus_1=-1
# b1=(x0_p-x1_p)*t_minus_1/t0_vec[m1]
# B.append(b1)
#
#
# #j=2,...,n
# def tk_xj_p_vec(j,mj_ind,B,S,T_ind):
#     """
#
#     :param j: step
#     :param mj_ind: index of mj in S
#     :param B: [b1,...,b_{j-1}]
#     :param S: |S|=n-j+1
#     :param T_ind: [m0,m1,...,m_{j-1}]
#     :return: t_{k}(x_{j}^{'}) values
#     , [t0(x_{j}^{'}), t_{1}(x_{j}^{'}),
#     ...,
#     t_{j-2}(x_{j}^{'}), t_{j-1}(x_{j}^{'})]
#     """
#
#     t_vec_tmp=[]
#     mj=S[mj_ind]
#
#     t_vec_tmp.append(t0_vec[mj])
#     xj_p=T_vals[mj]
#     #k=1,2,...,j-1
#     for k in range(1,j):
#         part1=B[k-1]*t_vec_tmp[k-1]
#         mk_minus_1=T_ind[k-1]
#         x_k_minus_1=T_vals[mk_minus_1]
#         if k==2:
#             part2=(xj_p-x_k_minus_1)*t_minus_1
#         else:
#             part2=(xj_p-x_k_minus_1)*t_vec_tmp[k-2]
#         t_vec_tmp.append(part1+part2)
#     return t_vec_tmp
#
#
# def last_2_elems_not_0(j,mj_ind,B,S,T_ind):
#     """
#
#     :param j: step
#     :param mj_ind: index of mj in S
#     :param B: [b1,...,b_{j-1}]
#     :param S: |S|=n-j+1
#     :param T_ind: [m0,m1,...,m_{j-1}]
#     :return: whether the last 2 elements of t_vec_tmp are not 0
#     """
#     t_vec_tmp=tk_xj_p_vec(j,mj_ind,B,S,T_ind)
#     val1=np.abs(t_vec_tmp[-2])
#     val2=np.abs(t_vec_tmp[-1])
#     if val1!=0 and val2!=0:
#         return True
#
#     else:
#         return False
#
# def iterate_mj_ind(j,B,S,T_ind):
#     """
#     find a valid mj_ind
#     :param j: step
#     :param B: [b1,...,b_{j-1}]
#     :param S: |S|=n-j+1
#     :param T_ind: [m0,m1,...,m_{j-1}]
#     :return: mj_ind such that bj can be computed
#     """
#     mj_ind=0
#     while last_2_elems_not_0(j,mj_ind,B,S,T_ind)==False:
#         mj_ind+=1
#     return mj_ind
#
#
# for j in range(2,n+1):
#     mj_ind=iterate_mj_ind(j,B,S,T_ind)
#     if mj_ind==n-j+1:
#         print(f"cannot find valid mj_ind for j={j}, mj_ind={mj_ind}")
#         exit(1)
#     else:
#         t_vec_tmp=tk_xj_p_vec(j,mj_ind,B,S,T_ind)
#         mj_minus1=T_ind[-1]
#
#         mj=S[mj_ind]
#
#         x_j_minus1_p=T_vals[mj_minus1]
#         x_j_p=T_vals[mj]
#
#         t_up_tmp=t_vec_tmp[-2]
#         t_down_tmp=t_vec_tmp[-1]
#
#         bj=(x_j_minus1_p-x_j_p)*t_up_tmp/t_down_tmp
#
#         B.append(bj)
#         S.pop(mj_ind)
#         T_ind.append(mj)
#
#
# print(f"B={B}")
# print(f"S={S}")
# print(f"T_ind={T_ind}")
#
#
# p_polynomial_vec=[]
# q_polynomial_vec=[]
#
# # initialization
# p_polynomial_vec.append(1)
# m0=T_ind[0]
# p_polynomial_vec.append(F1_vals[m0])
#
# q_polynomial_vec.append(0)
# q_polynomial_vec.append(1)
#
# #iteration for j=2,3,...,n
# for j in range(2,n+1):
#     m_j_minus1=T_ind[j-1]
#     x_j_minus1_p=T_vals[m_j_minus1]
#
#     #pj
#     pj_poly_tmp=B[j-1]*p_polynomial_vec[-1]+(T-x_j_minus1_p)*p_polynomial_vec[-2]
#     p_polynomial_vec.append(pj_poly_tmp)
#
#     #qj
#     qj_poly_tmp=B[j-1]*q_polynomial_vec[-1]+(T-x_j_minus1_p)*q_polynomial_vec[-2]
#     q_polynomial_vec.append(qj_poly_tmp)
#
#
# p_last=p_polynomial_vec[-1]
# q_last=q_polynomial_vec[-1]
# pprint(f"p_last: {expand(p_last)}")
# pprint(f"q_last: {expand(q_last)}")
#
#
# coeffs_p_last=poly(p_last,T).all_coeffs()
# coeffs_q_last=poly(q_last,T).all_coeffs()
# print(f"coeffs_p_last={coeffs_p_last}")
# print(f"coeffs_q_last={coeffs_q_last}")
#
# solutions_p=np.roots(coeffs_p_last)
# solutions_q=np.roots(coeffs_q_last)
#
# print(f"solutions_p={solutions_p}")
# print(f"solutions_q={solutions_q}")
#
#
# np_poly_q=np.poly1d(coeffs_q_last)
# print(f"{np_poly_q}")
# np_poly_dq=np_poly_q.deriv()
#
# which_T_ind=0
# T_num_c=solutions_q[which_T_ind]
# # T_num_c=1.13
# print(f"T_num_c={T_num_c}")
# p_val=np.polyval(coeffs_p_last,T_num_c)
# q_val=np_poly_q(T_num_c)
# q_deriv_val=np_poly_dq(T_num_c)
#
# print(f"p_val={p_val}")
# print(f"q_val={q_val}")
# print(f"q_deriv_val={q_deriv_val}")
#
# z=p_val/q_deriv_val
# print(f"z={z}")
#
# def r(x):
#     p_val=np.polyval(coeffs_p_last,x)
#     q_val=np.polyval(coeffs_q_last,x)
#
#     return p_val/q_val
#
#
# T_check_TT_alg_vec=np.linspace(TVec[0],T_vec_for_exact[-1],20)
#
# r_val_vec=r(T_check_TT_alg_vec)
# TT_exact_dlog_M_vec=func_dT_M(T_check_TT_alg_vec)/func_M(T_check_TT_alg_vec)
#
# plt.figure()
# plt.scatter(T_check_TT_alg_vec,r_val_vec,color="magenta",s=2,label="TT")
# plt.plot(T_check_TT_alg_vec,TT_exact_dlog_M_vec,color="blue", linestyle="--", linewidth=1,label="exact")
#
# plt.xlabel("$T$")
# plt.ylabel("$\partial_{T}M/M$")
# plt.title(f"N={N}")
# plt.legend(loc="best")
# plt.savefig(csvDataFolderRoot+f"/checkTT_M_N{N}.png")
# plt.close()