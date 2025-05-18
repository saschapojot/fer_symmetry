import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline, PchipInterpolator

from datetime import datetime

#this script uses data collapse for 6 vertex model
# for each pair of lengths


# 1) parameters
eps = 1.0         # energy unit
kB  = 1.0         # Boltzmann constant
xc=1/2
Tc  = -eps/(kB*np.log(xc))


def cN(N,t):
    """

    :param N: site number
    :param t: scaled temperature , not dimensionless
    :return: exact value of specific heat
    """
    x=1/2+1/2*t

    val=kB*(np.log(x))**2*2*N*(2*x)**N/(2+(2*x)**N)**2

    return val

#temperatures are on the right hand side of phase transition point tc=0
tVals=np.linspace(0,0.1,10)
#which norm
q=2

def diff_1_val(NA,NB,d,c,t_vec):
    """

    :param NA: data from NA by NA lattice are for interpolation
    :param NB: data from NB by NB lattice are for comparison
    :param d: trial value of d_exact
    :param c: trial value of c_exact
    :param t_vec: temperature points for interpolation
    :return:
    """
    #A, interpolate
    C_vec_for_A=np.array([cN(NA,t) for t in t_vec] )

    #rescaled specific heat for interpolation
    rescaled_C_vec_for_A=C_vec_for_A/NA**d
    #rescaled temperature for interpolation
    t_vec_rescaled_for_A=t_vec/NA**c
    func_A=PchipInterpolator(t_vec_rescaled_for_A,rescaled_C_vec_for_A)

    #rescaled temperature from B, in the range of t_vec_rescaled_for_A
    t_rsc_to_compare=[]
    C_vec_for_B=[]#true value for comparison, from B

    #B
    for ind, t_val in enumerate(t_vec):
        rsc_val=t_val/NB**c
        if rsc_val>=np.min(t_vec_rescaled_for_A) and rsc_val<= np.max(t_vec_rescaled_for_A):
            t_rsc_to_compare.append(rsc_val)
            C_vec_for_B.append(cN(NB,t_val))


    t_rsc_to_compare=np.array(t_rsc_to_compare)

    func_A_vals_on_B=func_A(t_rsc_to_compare)

    diff_val=np.linalg.norm(func_A_vals_on_B*NB**d-C_vec_for_B,ord=q)

    return diff_val**q

N_vec=[50,90,110,150]
d_val_vec=np.linspace(-3,3,50)
c_val_vec=np.linspace(-3,3,50)
result_matrix = np.zeros((len(d_val_vec), len(c_val_vec)))
tStart=datetime.now()

for i, d in enumerate(d_val_vec):
    for j, c in enumerate(c_val_vec):
        try:
            for ind0,N0 in enumerate(N_vec):
                for ind1,N1 in enumerate(N_vec):

                    val=diff_1_val(N0,N1,d,c,tVals)
                    result_matrix[i, j] +=val
        except Exception as e:
            print(f"Error at d={d}, c={c}: {e}")
            result_matrix[i, j] = np.nan


# Create heatmap
plt.figure(figsize=(10, 8))
plt.pcolormesh(c_val_vec, d_val_vec, result_matrix, cmap='viridis', shading='auto')
plt.colorbar(label='Difference value')
plt.xlabel('c value')
plt.ylabel('d value')
plt.title('Scaling collapse difference for various (d,c) parameters')

# Optionally find the minimum value and mark it
min_idx = np.unravel_index(np.nanargmin(result_matrix), result_matrix.shape)
min_d, min_c = d_val_vec[min_idx[0]], c_val_vec[min_idx[1]]

min_val = result_matrix[min_idx]

plt.plot(min_c, min_d, 'rx', markersize=10)
plt.annotate(f'Min: ({min_c:.3f}, {min_d:.3f})\nValue: {min_val:.6f}',
             (min_c, min_d), xytext=(10, 10), textcoords='offset points')

plt.tight_layout()
plt.savefig("all_pairs_min_6v.png")

print(f"Minimum difference value: {min_val:.6f} at d={min_d:.4f}, c={min_c:.4f}")

tEnd=datetime.now()

print("time: ",tEnd-tStart)