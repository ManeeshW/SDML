#!/bin/bash
# colors
B='\033[34m'
Z='\033[0m'

loops=1
for i in `seq 1 $loops`
do
    echo "No $i"
    echo "${B} Start generating dataset${Z}"
    python3 RunShipblenderCycles.py -N 2
    python3 time.py
done


#python3 timeStart.py  
## Update the packages of the system
#
#python3 RunShipblenderCycles.py -N 1000
#python3 time.py  


