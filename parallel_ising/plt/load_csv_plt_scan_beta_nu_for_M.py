import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime

import pandas as pd
#this script performs finite size scaling for different beta , nu values,
# and combine multiple plots as a matrix
# all N
if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

init_path=int(sys.argv[1])
row=sys.argv[2]



T_lower=1
T_upper=1.15
J_abs=1/2
Tc_exact=2*J_abs/np.log(1+np.sqrt(2))
print(f"Tc_exact={Tc_exact}")
Tc=1.134

def load_T_M_for_one_N(N):
    csvDataFolderRoot=f"../dataAll/N{N}/row{row}/csvOut_init_path{init_path}/"
    inCsvFile=csvDataFolderRoot+"/magnetization_plot.csv"
    df=pd.read_csv(inCsvFile)
    TVec=np.array(df["T"])
    MValsAll=np.array(df["M"])
    # M2ValsAll=MValsAll**2

    mask = (TVec > T_lower) & (TVec < T_upper)
    TInds = np.where(mask)[0]
    TToPlt=TVec[TInds]
    M_to_plot=MValsAll[TInds]
    return TToPlt, M_to_plot

NVec=[10,20,30,40]
T_vecs_all=[]#corresponding to each N
M_vec_all=[]#corresponding to each N

for N in NVec:
    TToPlt, M_to_plot=load_T_M_for_one_N(N)
    T_vecs_all.append(TToPlt)
    M_vec_all.append(M_to_plot)


def T_M_to_rescaled(TToPlt_one_N,M_to_plot_one_N,N,beta,nu):
    TToPlt_one_N=np.array(TToPlt_one_N)
    M_to_plot_one_N=np.array(M_to_plot_one_N)


    tau_one_N=(TToPlt_one_N-Tc)/Tc*N**(1/nu)
    M_rescaled_one_N=M_to_plot_one_N*N**(beta/nu)

    return M_rescaled_one_N,tau_one_N

# Set the Q value (grid size)
Q = 35  # Change this to 5, 10, 20, etc. as needed
beta_values=np.linspace(0.08,0.13,Q)
nu_values=np.linspace(0.5,2,Q)
# Adaptive figure sizing function
# Adaptive figure sizing function with larger base sizes
def get_figure_dimensions(Q):
    # Increase base size per subplot in inches
    base_size = 2.0  # Larger base size

    # For larger grids, reduce the base size to keep the figure manageable, but keep it larger than before
    if Q > 10:
        base_size = 1.8
    if Q > 15:
        base_size = 1.5  # Still 50% larger than previous smallest size

    # Add margins for labels, titles, etc.
    width = Q * base_size + 3  # +3 for margins
    height = Q * base_size + 3  # +3 for margins

    # Calculate appropriate spacing - more space for larger subplots
    spacing = max(0.15, 0.35 - 0.01 * Q)

    return width, height, spacing



# Create a color map for different N values
colors = ['b', 'r', 'g', 'purple']
# Get figure dimensions based on grid size
width, height, spacing = get_figure_dimensions(Q)
# Create figure and subplots with adaptive sizing
fig, axes = plt.subplots(Q, Q, figsize=(width, height))
plt.subplots_adjust(hspace=spacing, wspace=spacing)
t_plt_start=datetime.now()
for i, beta in enumerate(beta_values):
    for j, nu in enumerate(nu_values):
        ax = axes[i, j]
        # Plot rescaled data for each N
        for idx, N in enumerate(NVec):
            M_rescaled, tau = T_M_to_rescaled(T_vecs_all[idx], M_vec_all[idx], N, beta, nu)
            ax.scatter(tau, M_rescaled, s=10, c=colors[idx], label=f'N={N}')


        ax.text(0.05, 0.95, f'β={beta:.4f}, ν={nu:.4f}', transform=ax.transAxes,
                fontsize=8, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7))

        ax.tick_params(axis='both', which='major', labelsize=7)  # Larger tick labels
        ax.grid(True, alpha=0.3)
        # Add legend to the first plot only
        if i == 0 and j == 0:
            ax.legend()
        print(f"plotted ({i}, {j})")

t_plt_end=datetime.now()
print(f"time: ",t_plt_end-t_plt_start)
# Set common labels
fig.text(0.5, 0.04, r'Rescaled Temperature: (T-Tc)/Tc $\cdot N^{\frac{1}{\nu}}$', ha='center', fontsize=14)
fig.text(0.04, 0.5, r'Rescaled Magnetization: M $\cdot N^{\frac{\beta}{\nu}}$', va='center', rotation='vertical', fontsize=14)
fig.suptitle(r'Finite-Size Scaling: Data Collapse for Different $\beta$ and $\nu$ Values', fontsize=16)

# Save the figure
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'fss_matrix_plot_init{init_path}_row{row}.png', dpi=300, bbox_inches='tight')

outDir=f"../dataAll/row{row}/"
plt.savefig(outDir+f"/M_rescale_beta_nu_all_Q{Q}.png")
plt.close()