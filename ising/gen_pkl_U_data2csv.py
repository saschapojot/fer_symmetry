from pathlib import Path
from decimal import Decimal, getcontext
import shutil
import numpy as np
import pandas as pd
import os

#this script creates slurm bash files for running data2csv/pkl_U_data2csv.py separately

def format_using_decimal(value, precision=6):
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
N=64#unit cell number

init_path=0
which_row=0
startingFileIndSuggest=60
sweep_to_write=1000
lag=240
sweep_multiple=3

chunk_size = 12
T_vec1=[10+n*0.2 for n in range(0,6)]
T_vec2=[11.3 + n*0.02 for n in range(0,11)]#11.3 to 11.5
T_vec3=[20,50,80]
TVals=T_vec2
chunks = [TVals[i:i + chunk_size] for i in range(0, len(TVals), chunk_size)]


def contents_to_bash(chk_ind,T_ind,chunks):
    TStr=format_using_decimal(chunks[chk_ind][T_ind])
    contents=[
        "#!/bin/bash\n",
        "#SBATCH -n 1\n",
        "#SBATCH -N 1\n",
        "#SBATCH -t 0-60:00\n",
        "#SBATCH -p CLUSTER\n",
        "#SBATCH --mem=6GB\n",
        f"#SBATCH -o out_pkl_s_data2csv_{TStr}.out\n",
        f"#SBATCH -e out_pkl_s_data2csv_{TStr}.err\n",
        "cd /home/cywanag/data/hpc/cywanag/liuxi/Document/cppCode/fer_symmetry/ising\n",
        f"python3 -u ./data2csv/pkl_U_data2csv.py {N} {TStr} {init_path}  {which_row} {startingFileIndSuggest} {sweep_to_write} {lag} {sweep_multiple}\n"

    ]
    out_chunk=outPath+f"/chunk{chk_ind}/"
    Path(out_chunk).mkdir(exist_ok=True,parents=True)
    outBashName=out_chunk+f"/pkl_s_data2csv_T{TStr}.sh"
    with open(outBashName,"w+") as fptr:
        fptr.writelines(contents)

for chk_ind in range(0,len(chunks)):
    for T_ind in range(0,len(chunks[chk_ind])):
        contents_to_bash(chk_ind,T_ind,chunks)