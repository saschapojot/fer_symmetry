import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime

import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
#this script loads M2, with no scalings for M2
# for all N
#using variable lambda
if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]

# this script uses synthetic M2 data
# from asymptotic expansion

T_lower=0.8
T_upper=1.125
Tc=1.13

J=1/2
Tc_exact=2*J/np.log(1+np.sqrt(2))
def g(T,r):
    beta_T=1/T
    val=np.sinh(2*beta_T*J)

    return val**r


def x3_prime(T):
    up=g(T,-4)+1
    down=g(T,-4)-1

    return up/down

def M2_asymp(N,T):
    val0=(1-g(T,-4))**(1/4)
    val1=(2*np.pi*N**2)**(-1)

    val2=g(T,-4*N-4)

    val3=(g(T,-4)-1)**(-2)

    val4=1+7/4*N**(-1)*x3_prime(T)+1/32*N**(-2)*(117*x3_prime(T)**2-40)

    return val0*(1+val1*val2*val3*val4)

NVec=[100,200,400]
TVec=np.linspace(1.12,1.13,40)

def T_M_to_rescaled(TToPlt_one_N,M2_to_plot_one_N,N,beta,nu):
    TToPlt_one_N=np.array(TToPlt_one_N)
    M2_to_plot_one_N=np.array(M2_to_plot_one_N)
    lambda_one_N=N**(1/nu)*np.abs(TToPlt_one_N-Tc)/Tc
    tau_one_N=np.abs(TToPlt_one_N-Tc)/Tc
    M2_rescaled_to_plot_one_N=np.array(M2_to_plot_one_N)
    return lambda_one_N,M2_rescaled_to_plot_one_N,tau_one_N



beta=0.125/1
nu=1
# lambda_to_plot=np.linspace(50,100,20)
for ind,N in enumerate(NVec):
    TToPlt_one_N=TVec
    M2_to_plot_one_N=[M2_asymp(N,T) for T in TToPlt_one_N]
    lambda_one_N,M2_rescaled_to_plot_one_N,tau_one_N=T_M_to_rescaled(TToPlt_one_N,M2_to_plot_one_N,N,beta,nu)
    # plt.scatter(TToPlt_one_N,M2_rescaled_to_plot_one_N,label=f"N={N}",s=4)
    plt.scatter(lambda_one_N,M2_rescaled_to_plot_one_N**(2*beta/nu),label=f"N={N}",s=4)
    print(f"N={N}===========================")
    print(f"tau_one_N={tau_one_N}")
    print(f"lambda_one_N={lambda_one_N}")
# plt.axvline(x=N**(1/nu), color='r', linestyle='--')
# plt.xlim(5,70)
# plt.axvline(x=Tc_exact,color="red",linestyle="-.")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"scaled M^{2}$")
plt.legend(loc="best")
outDir=f"../dataAll/row{row}/"
plt.savefig(outDir+"/M2_synthetic_lambda.png")
plt.close()