import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime

import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
#this script loads M2, with rescalings
# for all N
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

def g(T,r):
    beta=1/T
    val=np.sinh(2*beta*J)

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

NVec=[50,100,150,200]
TVec=np.linspace(0.8,1.12,20)

def T_M_to_rescaled(TToPlt_one_N,M2_to_plot_one_N,N,beta,nu):
    TToPlt_one_N=np.array(TToPlt_one_N)
    M2_to_plot_one_N=np.array(M2_to_plot_one_N)

    eps_one_N=Tc/np.abs(TToPlt_one_N-Tc)*N**(-1/nu)
    tau_one_N=np.abs(TToPlt_one_N-Tc)/Tc
    M2_rescaled_one_N=M2_to_plot_one_N*np.abs(Tc/(Tc-TToPlt_one_N))**(2*beta)
    return eps_one_N,M2_rescaled_one_N,tau_one_N



beta=0.5
nu=1
for ind,N in enumerate(NVec):
    TToPlt_one_N=TVec
    M2_to_plot_one_N=[M2_asymp(N,T) for T in TToPlt_one_N]
    eps_one_N,M2_rescaled_one_N,tau_one_N=T_M_to_rescaled(TToPlt_one_N,M2_to_plot_one_N,N,beta,nu)
    plt.scatter(eps_one_N,M2_rescaled_one_N,label=f"N={N}",s=4)
    print(f"N={N}===========================")
    # print(f"tau_one_N={tau_one_N}")
    # print(f"eps_one_N={eps_one_N}")
    # print(f"M2_rescaled_one_N={M2_rescaled_one_N}")

    fit_num=5
    if len(eps_one_N)<fit_num:
        continue
    eps_fit=eps_one_N[:fit_num].reshape(-1,1)
    # print(f"eps_fit={eps_fit}")
    M2_rescaled_fit=M2_rescaled_one_N[:fit_num]
    print(f"M2_rescaled_fit={M2_rescaled_fit}")
    model= LinearRegression()
    model.fit(eps_fit,M2_rescaled_fit)
    # Get slope and intercept
    slope = model.coef_[0]
    intercept = model.intercept_

    # For prediction
    line_x = np.linspace(0, eps_one_N[:fit_num].max(), 100)  # 100 points for a smooth line
    line_x_2d = line_x.reshape(-1, 1)  # Reshape to 2D for prediction
    line_y = model.predict(line_x_2d)
    plt.plot(line_x, line_y,label=f"N={N}",linewidth=1)
    print(f"{line_y[0]}")







plt.xlim(0,0.1)

plt.xlabel(r"$\epsilon$")
plt.ylabel(r"scaled M^{2}$")
# plt.xscale("log")
# plt.yscale("log")
plt.legend(loc="best")
outDir=f"../dataAll/row{row}/"
plt.savefig(outDir+"/M2_synthetic_eps.png")
plt.close()