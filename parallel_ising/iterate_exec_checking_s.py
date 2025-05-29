import subprocess
from decimal import Decimal, getcontext
import signal
import sys
from pathlib import Path
from decimal import Decimal, getcontext

#this script runs exec_noChecking_s.py sequentially

def format_using_decimal(value, precision=4):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)


N=64#unit cell number

startingFileIndSuggest=5
init_path=0
which_row=0


T_vec1=[10+n*0.2 for n in range(0,6)]
T_vec2=[11.1]
T_vec3=[12+n*0.2 for n in range(0,6)]
TVals=T_vec2
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


# Register the signal handler for SIGINT
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
            ["python3", "exec_noChecking_s.py", TStr, str(N), str(startingFileIndSuggest), str(init_path), str(which_row)]
        )
        # Wait for the current subprocess to finish before proceeding to the next one.
        # current_process.wait()
        # current_process = None  # Reset the process variable

    except KeyboardInterrupt:
        pass



