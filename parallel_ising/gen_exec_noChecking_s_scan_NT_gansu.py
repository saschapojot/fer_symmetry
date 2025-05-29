from pathlib import Path
from decimal import Decimal, getcontext
import shutil
import numpy as np
import pandas as pd
import os
#this script creates slurm bash files for exec_noChecking_s.py

def format_using_decimal(value, precision=7):
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

num_parallel=24
startingFileIndSuggest=8
init_path=0
which_row=1
chunk_size = 100

b_vec=[1,2,3,4,5,6,7,8]

NVec_base=[5,6,7,8]

bN_vec=[]
for b in b_vec:
    for NBase in NVec_base:
        bN=b*NBase
        bN_vec.append(bN)
bN_vec=list(set(bN_vec))
print(f"bN_vec={bN_vec}")
T_vec1=[1.129,1.130,1.132,1.133,1.134,1.135,1.136]
TVals=T_vec1

default_flush_num=720
bN_T_pairs=[[bN,T] for bN in bN_vec for T in TVals]

chunks=[bN_T_pairs[i:i+chunk_size] for i in range(0,len(bN_T_pairs),chunk_size)]

def contents_to_bash(bN, T, file_index):
    NStr=format_using_decimal(bN)
    TStr=format_using_decimal(T)
    contents=[
        "#!/bin/bash\n",
        "#SBATCH -n 1\n",
        "#SBATCH -N 1\n",
        "#SBATCH -t 0-60:00\n",
        f"#SBATCH --cpus-per-task={num_parallel}\n",
        "#SBATCH -p lzicnormal\n",
        "#SBATCH --mem=10GB\n",
        f"#SBATCH -o out_exec_noChecking_s_N{NStr}_{TStr}.out\n",
        f"#SBATCH -e out_exec_noChecking_s_N{NStr}_{TStr}.err\n",
        "cd /public/home/hkust_jwliu_1/liuxi/Documents/cppCode/fer_symmetry/parallel_ising\n",
        f"python3 -u launch_one_run_dipole.py ./dataAll/N{NStr}/row{which_row}/T{TStr}/init_path{init_path}/run_T{TStr}_init_path{init_path}.mc.conf\n",
        f"numactl --interleave=all  ./run_mc ./dataAll/N{NStr}/row{which_row}/T{TStr}/init_path{init_path}/cppIn.txt\n"
    ]

    out_chunk=outPath+f"/chunk{file_index // chunk_size}/"
    Path(out_chunk).mkdir(exist_ok=True,parents=True)
    outBashName=out_chunk+f"/exec_checking_pol_N{NStr}_T{TStr}.sh"
    with open(outBashName,"w+") as fptr:
        fptr.writelines(contents)


# Process each pair with its index
for file_index, bN_T_pair in enumerate(bN_T_pairs):
    bN, T = bN_T_pair
    contents_to_bash(bN, T, file_index)

print(f"Generated {len(bN_T_pairs)} slurm files in {len(chunks)} chunks")