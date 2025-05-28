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


b_vec=[1,2,3,4,5,6,7,8]

NVec_base=[5,6,7,8]


which_row=1
# (1-eps)Tc, (1+eps)Tc
#Tc=1.13459265710651
# (1-eps)Tc=1.12324673053545
#(1+eps)Tc=1.14593858367758
T_vec1=[1.129,1.130,1.132,1.133,1.134,1.135,1.136]
TVals=T_vec1
default_flush_num=720
num_parallel=24
print(f"num_parallel={num_parallel}")
print(f"default_flush_num={default_flush_num}")
dataRoot="./dataAll/"

dataOutDir=dataRoot

effective_data_num_required=1000
sweep_to_write=500
sweep_multiple=6

in_param_file="./param.csv"
param_arr=pd.read_csv(in_param_file)
J=param_arr.iloc[which_row,0]
init_path_tot=1

print(f"J={J}")


J_Str=format_using_decimal(J)
TDirsAll=[]
TStrAll=[]
for k in range(0,len(TVals)):
    T=TVals[k]
    # print(T)

    TStr=format_using_decimal(T)
    TStrAll.append(TStr)

def contents_to_conf(k,which_init_ind,N_base,b):
    bN=b*N_base
    NStr=format_using_decimal(bN)
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
        f"N={NStr}\n",
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
        f"num_parallel={num_parallel}\n"
    ]
    outDir=dataOutDir+f"/N{NStr}/row{which_row}/T{TStrAll[k]}/init_path{which_init_ind}/"
    Path(outDir).mkdir(exist_ok=True,parents=True)
    outConfName=outDir+f"/run_T{TStrAll[k]}_init_path{which_init_ind}.mc.conf"
    with open(outConfName,"w+") as fptr:
        fptr.writelines(contents)

for k in range(0,len(TVals)):
    for N_base in NVec_base:
        for b in b_vec:
            contents_to_conf(k,0,N_base,b)
