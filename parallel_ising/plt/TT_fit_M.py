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
# for one N

if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()
N=int(sys.argv[1])
init_path=int(sys.argv[2])
row=sys.argv[3]

T_lower=0.9
T_upper=1.1
Tc=1.13
J=-1/2

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


T_return, M_return=load_M_for_one_N(N)

T_vals=T_return[::-1]
n=len(T_vals)-1
print(f"n={n}")
F1_vals=M_return[::-1]
print(f"F1_vals={F1_vals}")
outDir=f"../dataAll/row{row}/"

plt.figure()
plt.scatter(T_vals,F1_vals,color="black")
plt.xlabel("$T$")
plt.ylabel("$F_{1}(T)$")
plt.savefig(outDir+f"/N_{N}_F1.png")
plt.close()

B=[]
m0=0
T_ind=[m0]
S=[j for j in range(1,n+1)]
def t0(n):
    """

    :param n:
    :return: F1_vals[n]-F1_vals[0]
    """
    return  F1_vals[n]-F1_vals[0]

t0_vec=[t0(n) for n in range(0,len(T_vals))]


#j=1
m1_ind=0#index of m1 in S
m1=S[m1_ind]
# print(m1)
while np.abs(t0_vec[m1])==0 and m1_ind<len(S):
    m1_ind+=1
    m1=S[m1_ind]
# print(S)
S.pop(m1_ind)
# print(S)
T_ind.append(m1)
# print(T_ind)
x1_p=T_vals[m1]
x0_p=T_vals[m0]
t_minus_1=-1
b1=(x0_p-x1_p)*t_minus_1/t0_vec[m1]
B.append(b1)


#j=2,...,n
def tk_xj_p_vec(j,mj_ind,B,S,T_ind):
    """

    :param j: step
    :param mj_ind: index of mj in S
    :param B: [b1,...,b_{j-1}]
    :param S: |S|=n-j+1
    :param T_ind: [m0,m1,...,m_{j-1}]
    :return: t_{k}(x_{j}^{'}) values
    , [t0(x_{j}^{'}), t_{1}(x_{j}^{'}),
    ...,
    t_{j-2}(x_{j}^{'}), t_{j-1}(x_{j}^{'})]
    """

    t_vec_tmp=[]
    mj=S[mj_ind]

    t_vec_tmp.append(t0_vec[mj])
    xj_p=T_vals[mj]
    #k=1,2,...,j-1
    for k in range(1,j):
        part1=B[k-1]*t_vec_tmp[k-1]
        mk_minus_1=T_ind[k-1]
        x_k_minus_1=T_vals[mk_minus_1]
        if k==2:
            part2=(xj_p-x_k_minus_1)*t_minus_1
        else:
            part2=(xj_p-x_k_minus_1)*t_vec_tmp[k-2]
        t_vec_tmp.append(part1+part2)
    return t_vec_tmp


def last_2_elems_not_0(j,mj_ind,B,S,T_ind):
    """

    :param j: step
    :param mj_ind: index of mj in S
    :param B: [b1,...,b_{j-1}]
    :param S: |S|=n-j+1
    :param T_ind: [m0,m1,...,m_{j-1}]
    :return: whether the last 2 elements of t_vec_tmp are not 0
    """
    t_vec_tmp=tk_xj_p_vec(j,mj_ind,B,S,T_ind)
    val1=np.abs(t_vec_tmp[-2])
    val2=np.abs(t_vec_tmp[-1])
    if val1!=0 and val2!=0:
        return True

    else:
        return False

def iterate_mj_ind(j,B,S,T_ind):
    """
    find a valid mj_ind
    :param j: step
    :param B: [b1,...,b_{j-1}]
    :param S: |S|=n-j+1
    :param T_ind: [m0,m1,...,m_{j-1}]
    :return: mj_ind such that bj can be computed
    """
    mj_ind=0
    while last_2_elems_not_0(j,mj_ind,B,S,T_ind)==False:
        mj_ind+=1
    return mj_ind


for j in range(2,n+1):
    mj_ind=iterate_mj_ind(j,B,S,T_ind)
    if mj_ind==n-j+1:
        print(f"cannot find valid mj_ind for j={j}, mj_ind={mj_ind}")
        exit(1)
    else:
        t_vec_tmp=tk_xj_p_vec(j,mj_ind,B,S,T_ind)
        mj_minus1=T_ind[-1]

        mj=S[mj_ind]

        x_j_minus1_p=T_vals[mj_minus1]
        x_j_p=T_vals[mj]

        t_up_tmp=t_vec_tmp[-2]
        t_down_tmp=t_vec_tmp[-1]

        bj=(x_j_minus1_p-x_j_p)*t_up_tmp/t_down_tmp

        B.append(bj)
        S.pop(mj_ind)
        T_ind.append(mj)


print(f"B={B}")
print(f"S={S}")
print(f"T_ind={T_ind}")


# compute polynomials
T=symbols("T",cls=Symbol, real=True)

p_polynomial_vec=[]
q_polynomial_vec=[]

# initialization
p_polynomial_vec.append(1)
m0=T_ind[0]
p_polynomial_vec.append(F1_vals[m0])
# print(p_polynomial_vec)
# print(F1_vals)

q_polynomial_vec.append(0)
q_polynomial_vec.append(1)
#iteration for j=2,3,...,n
for j in range(2,n+1):
    m_j_minus1=T_ind[j-1]
    x_j_minus1_p=T_vals[m_j_minus1]

    #pj
    pj_poly_tmp=B[j-1]*p_polynomial_vec[-1]+(T-x_j_minus1_p)*p_polynomial_vec[-2]
    p_polynomial_vec.append(pj_poly_tmp)

    #qj
    qj_poly_tmp=B[j-1]*q_polynomial_vec[-1]+(T-x_j_minus1_p)*q_polynomial_vec[-2]
    q_polynomial_vec.append(qj_poly_tmp)


# pprint(p_polynomial_vec)
# pprint(q_polynomial_vec)
p_last=p_polynomial_vec[-1]
q_last=q_polynomial_vec[-1]
pprint(f"p_last: {expand(p_last)}")
pprint(f"q_last: {expand(q_last)}")

pprint(q_last.subs([(T,5)]).evalf())
coeffs_p_last=poly(p_last,T).all_coeffs()
coeffs_q_last=poly(q_last,T).all_coeffs()
print(f"coeffs_p_last={coeffs_p_last}")
print(f"coeffs_q_last={coeffs_q_last}")

def r(x):
    p_val=np.polyval(coeffs_p_last,x)
    q_val=np.polyval(coeffs_q_last,x)

    return p_val/q_val

solutions_q=np.roots(coeffs_q_last)
solutions_p=np.roots(coeffs_p_last)
print(f"solutions_p={solutions_p}")
print(f"solutions_q={solutions_q}")

which_T_ind=0
T_num_c=solutions_q[which_T_ind]
print(f"T_num_c={T_num_c}")

# p_T_num_c=p_last.subs([(T,T_num_c)])
#
# q_T_num_c=q_last.subs([(T,T_num_c)])
#
# print(f"p_T_num_c={p_T_num_c.evalf()}")
# print(f"q_T_num_c={q_T_num_c.evalf()}")