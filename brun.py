import subprocess

subprocess.check_output(["python3 BlenderCyclesRender/RunShipblenderCycles.py -N {}".format(1)],shell=True)