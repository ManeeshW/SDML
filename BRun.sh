#!/bin/bash
SUBNAME="2"

loops=2
for i in `seq 1 $loops`
do
    echo "No $i"
    echo "Start generating dataset"
    python3 BlenderCyclesRender/RunShipblenderCycles.py -N 2 --ss $SUBNAME
#    python3 BlenderCyclesRender/time.py
done


