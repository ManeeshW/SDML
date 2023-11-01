#!/bin/bash
SUBNAME=1

loops=73
for i in `seq 1 $loops`
do
    python3 BlenderCyclesRender/RunShipblenderCycles.py -N $i 
#    python3 BlenderCyclesRender/time.py
done
