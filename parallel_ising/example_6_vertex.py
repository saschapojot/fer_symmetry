import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline, PchipInterpolator


#this script uses data collapse for 6 vertex model

# 1) parameters
eps = 1.0         # energy unit
kB  = 1.0         # Boltzmann constant
xc=1/2
Tc  = -eps/(kB*np.log(xc))


def cN(N,t):
    x=1/2+1/2*t

    val=kB*(np.log(x))**2*2*N*(2*x)**N/(2+(2*x)**N)**2

    return val

tVals=np.linspace(0,0.1,10)
d_exact=1
c_exact=-1
def rescaled_values(N,t_vec):
    """

    :param N:
    :param t_vec:
    :return: rescaled t and C
    """
    C_vec=[cN(N,t) for t in t_vec]

    C_vec=np.array(C_vec)

    C_vec_rescaled=C_vec/N**d_exact

    t_vec_rescaled=t_vec/N**c_exact

    return t_vec_rescaled,C_vec_rescaled,C_vec


N_vec=[30,50,70,90,110,130,150]
plt.figure()
for N in N_vec:
    t_vec_rescaled,C_vec_rescaled,_=rescaled_values(N,tVals)
    plt.scatter(t_vec_rescaled,C_vec_rescaled,label=f"N={N}")

plt.xlabel("rescaled t")
plt.ylabel("rescaled c")
plt.yscale("log")
plt.legend(loc="best")
plt.savefig("6v.png")
plt.close()

tVals=np.linspace(-0.1,0.1,20)
c_per_site_xc=1/4*np.log(2)**2
print(f"c_per_site_xc={c_per_site_xc}")
#plot curves without rescaling

for N in N_vec:
    _,_,C_vec=rescaled_values(N,tVals)
    C_vec=np.array(C_vec)
    C_per_site_vec=C_vec/N
    spl_func=PchipInterpolator(tVals,C_per_site_vec)
    fig, ax = plt.subplots()
    plt.scatter(tVals,C_per_site_vec)
    plt.plot(tVals,spl_func(tVals),color="blue")
    ax.axhline(y=c_per_site_xc, color='r', linestyle='--')
    plt.xlabel("$t$")
    plt.ylabel("$c$ per site")
    plt.title(f"N={N}")
    plt.savefig(f"6v_N{N}.png")
    plt.close()

q=2

def diff_1_val(N0_ind,N1_ind,d,c,t_vec):
    p=N0_ind
    j=N1_ind

    N0=N_vec[p]
    N1=N_vec[j]
    print(f"N0={N0}")
    print(f"N1={N1}")

    C_vec_for_p=np.array([cN(N0,t) for t in t_vec])#for interpolation
    # C_vec_for_j=np.array([cN(N1,t) for t in t_vec])# for comparison

    #interpolate using set p: N0
    rescaled_C_vec_for_p=C_vec_for_p/N0**d
    t_vec_rescaled_for_p=t_vec/N0**c
    E_p=PchipInterpolator(t_vec_rescaled_for_p,rescaled_C_vec_for_p)

    t_rsc_to_compare=[]
    rescaled_C_vec_for_j=[]
    #set j: N1
    for ind, t_val in enumerate(t_vec):
        rsc_val=t_val/N1**c
        if rsc_val>=np.min(t_vec_rescaled_for_p) and rsc_val<= np.max(t_vec_rescaled_for_p):
            t_rsc_to_compare.append(rsc_val)
            rescaled_C_vec_for_j.append(cN(N1,t_val)/N1**d )



    # print(f"t_vec_rescaled_for_p={t_vec_rescaled_for_p}")

    # print(f"t_rsc_to_compare={t_rsc_to_compare}")
    t_rsc_to_compare=np.array(t_rsc_to_compare)
    E_p_vals_on_t_rsc_j=E_p(t_rsc_to_compare)

    print(f"E_p_vals_on_t_rsc_j={E_p_vals_on_t_rsc_j}")
    print(f"rescaled_C_vec_for_j={rescaled_C_vec_for_j}")
    diff_val=np.linalg.norm(E_p_vals_on_t_rsc_j-rescaled_C_vec_for_j,ord=q)
    print(f"diff_val={diff_val}")
    return diff_val**(1/q)




N0_ind=0
N1_ind=1
d_tmp=0.5
c_tmp=-0.1
nm=diff_1_val(N0_ind,N1_ind,d_tmp,c_exact,tVals)

print(f"nm={nm}")