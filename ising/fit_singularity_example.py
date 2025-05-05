import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sympy import *
from sklearn.linear_model import LinearRegression

##this script is an example of using TT algorithm
#to fit a singularity

alpha=2
beta=1.2
gamma=1.42
Tc=1.4

# def f(T):
#     val=alpha*(T-Tc)**(-gamma)+beta
#     return val


def F1(T):
    numerator_val=-gamma*alpha*(T-Tc)**(-gamma-1)

    denominator_val=alpha*(T-Tc)**(-gamma)+beta

    return numerator_val/denominator_val


# interpolation T points

T_vals=Tc+np.array([0.005,0.006,0.01,0.02,0.03,0.04,0.05,0.1])
T_vals=T_vals[::-1]
n=len(T_vals)-1
print(f"n={n}")
# t_func_table=np.zeros((n+2,n))
# f_vals=[f(T) for T in T_vals]
# d log vals
F1_vals=[F1(T) for T in T_vals]
print(f"F1_vals={F1_vals}")
plt.figure()
plt.scatter(T_vals,F1_vals,color="black")
plt.xlabel("$T$")
plt.ylabel("$F_{1}(T)$")
plt.savefig("F1.png")
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

solutions=np.roots(coeffs_q_last)

print(f"solutions={solutions}")

T_num_c=solutions[2]
print(f"T_num_c={T_num_c}")


# p_val=np.polyval(coeffs_p_last,T_num_c)
# qprime_coeffs =np.polyder(coeffs_q_last)
# q_val=np.polyval(coeffs_q_last,T_num_c)
# print(f"q_val={q_val}")
# qprime_val =np.polyval(qprime_coeffs,T_num_c)
# residue = p_val / qprime_val
#
#
# print("p(x0) =", p_val)
# print("q'(x0) =", qprime_val)
# print("Residue at x0 =", residue)
#
inv_T_vals=np.array([1/(T-T_num_c) for T in T_vals]).reshape(-1, 1)
# Create and fit the model
model = LinearRegression()
model.fit(inv_T_vals, F1_vals)
# Extract coefficients
slope = model.coef_[0]
intercept = model.intercept_
print("Slope:", slope)
print("Intercept:", intercept)