from pathlib import Path
from decimal import Decimal, getcontext

import numpy as np
import pandas as pd


def format_using_decimal(value, precision=4):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)


N=128#unit cell number
N0=N
N1=N
which_row=0


T_vec1=[10+n*0.2 for n in range(0,6)]
T_vec2=[11.3 + n*0.02 for n in range(0,11)]#11.3 to 11.5
T_vec3=[20,50,80]
TVals=T_vec2
default_flush_num=120

dataRoot="./dataAll/"

dataOutDir=dataRoot

effective_data_num_required=1000
sweep_to_write=1000
sweep_multiple=3
in_param_file="./param.csv"
param_arr=pd.read_csv(in_param_file)
J=param_arr.iloc[which_row,0]
init_path_tot=1
print(f"J={J}")
print(f"N={N}")
print(f"TVals={TVals}")
print(f"len(TVals)={len(TVals)}")
J_Str=format_using_decimal(J)

TDirsAll=[]
TStrAll=[]
NStr=format_using_decimal(N)
for k in range(0,len(TVals)):
    T=TVals[k]
    # print(T)

    TStr=format_using_decimal(T)
    TStrAll.append(TStr)


def contents_to_conf(k,which_init_ind):
    contents=[
        "#This is the configuration file for 2d Ising mc computations\n",
        "\n" ,
        "#parameters\n",
        "#Temperature\n",
        "T="+TStrAll[k]+"\n",
        "#which init path\n",
        f"init_path={which_init_ind}\n",
        "\n",
        f"J={J_Str}\n",
        "\n",
        f"row={which_row}\n"
        "\n",
        f"N={NStr}\n"
        "\n",
        "#this is the data number in each pkl file, i.e., in each flush\n"
        f"sweep_to_write={sweep_to_write}\n",
        "#within each flush,  sweep_to_write*sweep_multiple mc computations are executed\n",
        "\n",
        f"default_flush_num={default_flush_num}\n",
        "\n",
        "#the configurations of the system are saved to file if the sweep number is a multiple of sweep_multiple\n",
        "\n",
        f"sweep_multiple={sweep_multiple}\n",
        ]

    outDir=dataOutDir+f"/N{NStr}/row{which_row}/T{TStrAll[k]}/init_path{which_init_ind}/"
    Path(outDir).mkdir(exist_ok=True,parents=True)
    outConfName=outDir+f"/run_T{TStrAll[k]}_init_path{which_init_ind}.mc.conf"
    with open(outConfName,"w+") as fptr:
        fptr.writelines(contents)


for k in range(0,len(TVals)):
    for j in range(0,init_path_tot):
        contents_to_conf(k,j)