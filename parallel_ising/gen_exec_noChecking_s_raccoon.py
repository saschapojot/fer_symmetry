from pathlib import Path
from decimal import Decimal, getcontext
import shutil
import numpy as np
import pandas as pd
import os
#this script creates slurm bash files for exec_noChecking_s.py
def format_using_decimal(value, precision=4):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)

outPath="./bashFiles_dipole_exec_noChecking/"
if os.path.isdir(outPath):
    shutil.rmtree(outPath)


Path(outPath).mkdir(exist_ok=True,parents=True)
N=64 #unit cell number
startingFileIndSuggest=8
init_path=0
which_row=1
chunk_size = 100

###for row 0
# T_vec1=[10+n*0.2 for n in range(0,6)]
# T_vec2=[11.3 + n*0.02 for n in range(0,11)]#11.3 to 11.5
# T_vec3=[20,50,80]
# TVals=T_vec2
#for row 1
T_vec1=[0.5+0.1*n for n in range(0,6)]+[0.5+0.1*n for n in range(8,16)]
T_vec2=[1.12+0.02*n for n in range(0,5)]#1.12, 1.14, 1.16, 1.18, 1.2
T_vec3=[1.13,1.15,1.17,1.19]
T_vec4=[1.21,1.22,1.23,1.24,1.25]
T_vec_tmp=[2.1]
TVals=T_vec_tmp
chunks = [TVals[i:i + chunk_size] for i in range(0, len(TVals), chunk_size)]

def contents_to_bash(chk_ind,T_ind,chunks):
    TStr=format_using_decimal(chunks[chk_ind][T_ind])
    contents=[
        "#!/bin/bash\n",
        "#SBATCH -n 1\n",
        "#SBATCH -N 2\n",
        "#SBATCH -t 0-60:00\n",
        "#SBATCH -p CLUSTER\n",
        "#SBATCH --mem=8GB\n",
        "#SBATCH --cpus-per-task=30\n",
        f"#SBATCH -o out_exec_noChecking_s_{TStr}.out\n",
        f"#SBATCH -e out_exec_noChecking_s_{TStr}.err\n",
        "cd /home/cywanag/data/hpc/cywanag/liuxi/Document/cppCode/fer_symmetry/parallel_ising\n",
        f"python3 -u launch_one_run_dipole.py ./dataAll/N{N}/row{which_row}/T{TStr}/init_path{init_path}/run_T{TStr}_init_path{init_path}.mc.conf\n",
        f"./run_mc ./dataAll/N{N}/row{which_row}/T{TStr}/init_path{init_path}/cppIn.txt\n"

        ]
    out_chunk=outPath+f"/chunk{chk_ind}/"
    Path(out_chunk).mkdir(exist_ok=True,parents=True)
    outBashName=out_chunk+f"/exec_checking_pol_T{TStr}.sh"
    with open(outBashName,"w+") as fptr:
        fptr.writelines(contents)


for chk_ind in range(0,len(chunks)):
    for T_ind in range(0,len(chunks[chk_ind])):
        contents_to_bash(chk_ind,T_ind,chunks)