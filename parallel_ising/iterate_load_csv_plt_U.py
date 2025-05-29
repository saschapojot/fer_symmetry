import subprocess
from decimal import Decimal, getcontext
import signal
import sys
from pathlib import Path
from decimal import Decimal, getcontext
import glob
import os

import re
import numpy  as np
#this script runs plt/load_csv_plt_U.py sequentially

def format_using_decimal(value, precision=4):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)


row=1

init_path=0


data_N_root="./dataAll/"

N_dir_vec=[]
N_vals=[]

for NDir in glob.glob(data_N_root+"/N*"):
    matchN=re.search(r"N(\d+)",NDir)
    if matchN:
        N_dir_vec.append(NDir)
        N_vals.append(int(matchN.group(1)))

# print(N_dir_vec)
# print(N_vals)

sortedInds=np.argsort(N_vals)

sorted_N_vals=[N_vals[ind] for ind in sortedInds]

sorted_N_dirs=[N_dir_vec[ind] for ind in sortedInds]

# Global variable to keep track of the currently running subprocess.
current_process = None

def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) signal to gracefully terminate the subprocess."""
    global current_process
    print("\nReceived Ctrl+C. Terminating gracefully...")
    if current_process is not None:
        current_process.terminate()
        try:
            current_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            current_process.kill()
    sys.exit(0)


#Register the signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)

NStrAll=[]
for j in range(0,len(sorted_N_vals)):
    N=sorted_N_vals[j]

    NStr=format_using_decimal(N)
    NStrAll.append(NStr)


# print(NStrAll)

for NStr in NStrAll:
    try:
        print(f"Executing for N = {NStr} ...")
        # Change to plt directory
        # os.chdir("./plt")
        current_process = subprocess.Popen(
            ["python3", "./plt/load_csv_plt_U.py", f"{NStr}", f"{init_path}", f"{row}"]
        )
        # Wait for the current subprocess to finish before proceeding to the next one.
        current_process.wait()
        current_process = None  # Reset the process variable
    except KeyboardInterrupt:
        # This block may catch Ctrl+C before the signal handler is invoked.
        print("\nKeyboard interrupt received. Terminating subprocess...")
        if current_process is not None:
            current_process.terminate()
            current_process.wait()

        sys.exit(0)