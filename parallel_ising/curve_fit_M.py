from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


J=-Rational(1,2)
J_np=-1/2
T=symbols("T",cls=Symbol,real=True)
beta=1/T
Tc_exact=-2*J_np/np.log(1+np.sqrt(2))
Tc=1.13

M_exact=(1-(sinh(2*beta*J)**(-4)))**(Rational(1,8))

dT_M_exact=diff(M_exact,T)
dlog_M_exact=dT_M_exact/M_exact

func_dT_M=lambdify(T,dT_M_exact,"numpy")
func_M=lambdify(T,M_exact,"numpy")
func_dlog_M=lambdify(T,dlog_M_exact,"numpy")

T_vec=np.linspace(0.8*Tc,Tc*0.9,100)
T_2_fit_vec=np.linspace(0.85*Tc,Tc*0.9,20)

M_exact_vec=func_M(T_vec)
dT_M_exact_vec=func_dT_M(T_vec)

M_2_fit_vec=func_M(T_2_fit_vec)


dlog_M_exact_vec=dT_M_exact_vec/M_exact_vec

dlog_fit_M=func_dT_M(T_2_fit_vec)/func_M(T_2_fit_vec)


def y_func_rational(T,alpha,D):
    val_up=-1-2*D*(Tc-T)

    val_down=Tc-T+D*(Tc-T)**2

    return alpha*val_up/val_down

def y_func_rational_N3(T,beta,b2,b3):
    val_up=-1-2*b2*(Tc-T)-3*b3*(Tc-T)**2

    val_down=Tc-T+b2*(Tc-T)**2+b3*(Tc-T)**3

    return beta*val_up/val_down
def y_func_pow(T,beta,A1,A2,A3):
    x=Tc-T
    val=A1*x+A2*x**2+A3*x**3
    return val**beta

def y_func_log(T,beta,A1,A2,A3):
    x=Tc-T
    val=A1*x+A2*x**2+A3*x**3
    return beta*np.log(val)

# Perform the curve fit
popt, pcov = curve_fit(y_func_rational, T_2_fit_vec, dlog_fit_M)


alpha_fit,D_fit=popt
print(f"alpha_fit={alpha_fit}, D_fit={D_fit}")

popt_N3,pcov_N3=curve_fit(y_func_rational_N3,T_2_fit_vec,dlog_fit_M)

perr_N3 = np.sqrt(np.diag(pcov_N3))  # Parameter uncertainties
beta_fit_N3,b2_fit_N3,b3_fit_N3=popt_N3
print(f"beta_fit_N3={beta_fit_N3} ± {perr_N3[0]:.4f}, b2_fit_N3={b2_fit_N3} ± {perr_N3[1]:.4f}, b3_fit_N3={b3_fit_N3} ± {perr_N3[2]:.4f}")

# popt_pow,pcov_pow=curve_fit(y_func_pow,T_2_fit_vec,M_2_fit_vec)
# beta_fit_pow,A1_fit_pow,A2_fit_pow,A3_fit_pow=popt_pow
# print(f"beta_fit_pow={beta_fit_pow}, A1_fit_pow={A1_fit_pow}, A2_fit_pow={A2_fit_pow},A3_fit_pow={A3_fit_pow}")

popt_log,pcov_log=curve_fit(y_func_log,T_2_fit_vec,np.log(M_2_fit_vec))
beta_fit_log,A1_fit_log,A2_fit_log,A3_fit_log=popt_log
print(f"beta_fit_log={beta_fit_log}, A1_fit_log={A1_fit_log}, A2_fit_log={A2_fit_log},A3_fit_log={A3_fit_log}")

T_2_fit_plot=np.linspace(T_2_fit_vec[0],T_2_fit_vec[-1],30)
y_fit = y_func_rational(T_2_fit_plot, *popt)

plt.figure()
plt.plot(T_vec,dlog_M_exact_vec,color="blue",linewidth=1,label="exact")
plt.scatter(T_2_fit_plot,y_fit,label="fit")
plt.xlabel("T")
plt.ylabel("M")
# plt.xscale("log")
# plt.yscale("log")
plt.legend(loc="best")
plt.savefig("dlog_M.png")
plt.close()