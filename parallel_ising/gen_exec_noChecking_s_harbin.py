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

startingFileIndSuggest=10
init_path=0
which_row=1
chunk_size = 100
#T_vec1: no phase transition

T_vec1=[0.5+0.1*n for n in range(0,6)]+[0.5+0.1*n for n in range(8,16)]

#the following 3 vectors are in phase transition regime
T_vec2=[1.12+0.02*n for n in range(0,5)]#1.12, 1.14, 1.16, 1.18, 1.2
T_vec3=[1.13,1.15,1.17,1.19]
T_vec4=[1.21,1.22,1.23,1.24,1.25]
T_vec5=[1.141+0.001*n for n in range(0,9)]
T_vec6=[1.151+0.001*n for n in range(0,9) ]

#the following 2 vectors are near Tc, <Tc
T_vec7=[0.95,0.96,0.97,0.98,0.99]
T_vec8=[1.01+n*0.01 for n in range(0,12)]#1.01,1.02,...,1.12
TVals=T_vec7+T_vec8+T_vec4+T_vec3

N=300#unit cell number
num_parallel=64

chunks = [TVals[i:i + chunk_size] for i in range(0, len(TVals), chunk_size)]

def contents_to_bash(chk_ind,T_ind,chunks):
    TStr=format_using_decimal(chunks[chk_ind][T_ind])
    contents=[
        "#!/bin/bash\n",
        "#SBATCH -n 1\n",
        "#SBATCH -N 1\n",
        "#SBATCH -t 0-60:00\n",
        f"#SBATCH --cpus-per-task={num_parallel}\n",
        "#SBATCH -p hebhcnormal01\n",
        "#SBATCH --mem=10GB\n",
        f"#SBATCH -o out_exec_noChecking_s_{TStr}.out\n",
        f"#SBATCH -e out_exec_noChecking_s_{TStr}.err\n",
        f"export OMP_NUM_THREADS={num_parallel}\n",
        f"export GOMP_CPU_AFFINITY=\"0-{num_parallel-1}\"\n"
        "cd /public/home/hkust_jwliu_1/liuxi/Document/cppCode/fer_symmetry/parallel_ising\n",
        f"python3 -u launch_one_run_dipole.py ./dataAll/N{N}/row{which_row}/T{TStr}/init_path{init_path}/run_T{TStr}_init_path{init_path}.mc.conf\n",
        f"numactl --interleave=all  ./run_mc ./dataAll/N{N}/row{which_row}/T{TStr}/init_path{init_path}/cppIn.txt\n"
        ]
    out_chunk=outPath+f"/chunk{chk_ind}/"
    Path(out_chunk).mkdir(exist_ok=True,parents=True)
    outBashName=out_chunk+f"/exec_checking_pol_T{TStr}.sh"
    with open(outBashName,"w+") as fptr:
        fptr.writelines(contents)


for chk_ind in range(0,len(chunks)):
    for T_ind in range(0,len(chunks[chk_ind])):
        contents_to_bash(chk_ind,T_ind,chunks)