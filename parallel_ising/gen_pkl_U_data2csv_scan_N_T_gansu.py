from pathlib import Path
from decimal import Decimal, getcontext
import shutil
import numpy as np
import pandas as pd
import os


#this script creates slurm bash files for running data2csv/pkl_U_data2csv.py separately

def format_using_decimal(value, precision=7):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)

outPath="./bashFiles_pkl_U_data2csv/"
if os.path.isdir(outPath):
    shutil.rmtree(outPath)
Path(outPath).mkdir(exist_ok=True,parents=True)

b_vec=[1,2,3,4,5,6,7,8]

NVec_base=[5,6,7,8]

bN_vec=[]
for b in b_vec:
    for NBase in NVec_base:
        bN=b*NBase
        bN_vec.append(bN)
bN_vec=list(set(bN_vec))
print(f"bN_vec={bN_vec}")
which_row=1
init_path=0
startingFileIndSuggest=30
sweep_to_write=500
sweep_multiple=6
lag=150
T_vec1=[1.129,1.130,1.132,1.133,1.134,1.135,1.136]
TVals=T_vec1

chunk_size = 100

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
        "#SBATCH -p lzicnormal\n",
        "#SBATCH --mem=6GB\n",
        f"#SBATCH -o out_pkl_U_data2csv_N{NStr}_{TStr}.out\n",
        f"#SBATCH -e out_pkl_U_data2csv_N{NStr}_{TStr}.err\n",
        "cd  /public/home/hkust_jwliu_1/liuxi/Documents/cppCode/fer_symmetry/parallel_ising\n",
        f"python3 -u ./data2csv/pkl_U_data2csv.py {NStr} {TStr} {init_path}  {which_row} {startingFileIndSuggest} {sweep_to_write} {lag} {sweep_multiple}\n"

    ]

    out_chunk=outPath+f"/chunk{file_index // chunk_size}/"
    Path(out_chunk).mkdir(exist_ok=True,parents=True)
    outBashName=out_chunk+f"/pkl_U_data2csv_N{NStr}_T{TStr}.sh"
    with open(outBashName,"w+") as fptr:
        fptr.writelines(contents)


# Process each pair with its index
for file_index, bN_T_pair in enumerate(bN_T_pairs):
    bN, T = bN_T_pair
    contents_to_bash(bN, T, file_index)


print(f"Generated {len(bN_T_pairs)} slurm files in {len(chunks)} chunks")