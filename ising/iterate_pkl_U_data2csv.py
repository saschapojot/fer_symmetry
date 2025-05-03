import subprocess
from decimal import Decimal, getcontext
import signal
import sys
from pathlib import Path
from decimal import Decimal, getcontext
#this script runs data2csv/pkl_U_data2csv.py sequentially

def format_using_decimal(value, precision=4):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)
N=16 #unit cell number
row=0
startingFileIndSuggest=5
init_path=0


T_start=1.5
T_end=16.5
T_step=0.5
number=int((T_end-T_start)/T_step)
TVals=[T_start+n*T_step for n in range(0,number+1)]



TStrAll=[]
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
for k in range(0,len(TVals)):
    T=TVals[k]
    # print(T)

    TStr=format_using_decimal(T)
    TStrAll.append(TStr)


# Execute the subprocess sequentially.

for TStr in TStrAll:
    try:
        print(f"Executing for T = {TStr} ...")
        current_process = subprocess.Popen(
            ["python3", "./data2csv/pkl_U_data2csv.py", f"{N}", f"{TStr}", f"{init_path}", str(row)]
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