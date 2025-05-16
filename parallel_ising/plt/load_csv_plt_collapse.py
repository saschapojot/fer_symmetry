import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import UnivariateSpline
#this script uses data collapse in paper  A measure of data collapse for scaling
# by Somendra M Bhattacharjee
# J. Phys. A: Math. Gen. 34 (2001) 6375–6380
if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]
Tc=1.12
def load_data_one_N(N):
    """

    :param N:
    :return: rescaled temperature, chi
    """
    csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
    inCsvFile=csvDataFolderRoot+"/magnetization_plot.csv"
    df=pd.read_csv(inCsvFile)
    TVec=np.array(df["T"])
    mask = (TVec > 1.12) & (TVec <2)

    TInds = np.where(mask)[0]
    TToPlt=TVec[TInds]
    chi_each_site_all=np.array(df["chi_each_site"])
    chi_to_plot=chi_each_site_all[TInds]
    t=np.abs(TToPlt-Tc)/Tc

    return t, chi_to_plot


NVec=[32,64,128]
N0=NVec[0]
N1=NVec[1]
N2=NVec[2]


t0_vec,chi0_vec=load_data_one_N(N0)
t1_vec,chi1_vec=load_data_one_N(N1)
t2_vec,chi2_vec=load_data_one_N(N2)

data=[]
data.append((N0,t0_vec,chi0_vec))
data.append((N1,t1_vec,chi1_vec))
data.append((N2,t2_vec,chi2_vec))


def compute_Pb(data, d, c, q=1):
    """
    Compute the data collapse measure P_b.

    Parameters:
    - data: List of tuples (L, t_array, m_array) for each dataset.
    - d, c: Scaling exponents.
    - q: Exponent for the residual norm (default: L1 norm).

    Returns:
    - P_b: Collapse measure.
    - N_over: Number of overlapping points.
    """
    residuals = []
    N_over = 0

    for p_idx, (L_p, t_p, m_p) in enumerate(data):
        # Rescale the reference set p: x_p = t_p / L_p^c, y_p = m_p / L_p^d
        x_p = t_p / (L_p ** c)
        y_p = m_p / (L_p ** d)

        # Create interpolator for set p (avoid extrapolation)
        # interp_func = interp1d(x_p, y_p, kind='cubic', bounds_error=False, fill_value=np.nan)
        interp_func = UnivariateSpline(x_p, y_p, k=3, s=len(x_p))  # s controls smoothness
        for j_idx, (L_j, t_j, m_j) in enumerate(data):
            if j_idx == p_idx:
                continue  # Skip self-comparison

            # Rescale set j: x_j = t_j / L_j^c, y_j = m_j / L_j^d
            x_j = t_j / (L_j ** c)
            y_j = m_j / (L_j ** d)

            # Find overlapping points (x_j within x_p's range)
            x_min_p, x_max_p = np.nanmin(x_p), np.nanmax(x_p)
            mask =[True]*len(x_p) #(x_j >= x_min_p) & (x_j <= x_max_p)
            # mask=(x_j >= x_min_p) & (x_j <= x_max_p)
            x_j_over = x_j[mask]
            y_j_over = y_j[mask]

            if len(x_j_over) == 0:
                continue  # No overlap

            # Interpolate y_p values at x_j_over and compute residuals
            y_p_interp = interp_func(x_j_over)
            valid = ~np.isnan(y_p_interp)
            residuals.extend(np.abs(y_j_over[valid] - y_p_interp[valid]))
            N_over += np.sum(valid)
    print(f"N_over={N_over}")
    if N_over==0:
        return np.inf, 0
    P_b = (np.sum(np.array(residuals) ** q) / N_over) ** (1/q)
    return P_b, N_over

def optimize_exponents(data, d_guess=1.0, c_guess=1.0, q=2):
    """
    Find optimal (d, c) that minimize P_b.

    Returns:
    - result: Optimization result object from scipy.
    """
    def objective(params):
        d, c = params
        P_b, _ = compute_Pb(data, d, c, q)
        return P_b

    result = minimize(
        objective,
        x0=[d_guess, c_guess],
        method='Nelder-Mead',  # Works well for low-dimensional problems
        options={'maxiter': 1000}
    )
    return result


# Optimize
result_noisy = optimize_exponents(data, d_guess=0.8, c_guess=0.6)
print(f"Noisy data optimal exponents: d = {result_noisy.x[0]:.3f}, c = {result_noisy.x[1]:.3f}")
print(f"Minimum P_b = {result_noisy.fun:.3e}")
def plot_collapse(data, d, c, title):
    plt.figure()
    for L, t, m in data:
        x = t / (L ** c)
        y = m / (L ** d)
        plt.scatter(x, y, label=f"L={L}")
    plt.xlabel(f"$t / L^{c}$")
    plt.ylabel(f"$m / L^{d}$")
    plt.legend()
    plt.title(title)
    plt.savefig(f"{title}.png")


# Plot after optimization
plot_collapse(data, d=result_noisy.x[0], c=result_noisy.x[1], title="After optimization")