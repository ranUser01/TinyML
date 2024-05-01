#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Please provide the Python script file as an argument."
    exit 1
fi

python_script=$1
current_datetime=$(date +"%Y%m%d%H%M%S")  
result_file="results/$(basename "$python_script" .py)_$current_datetime.txt"

python "$python_script" &
pid=$!

while true; do
    cpu_util=$(top -b -n 1 -p $pid | awk 'NR>7 { sum += $9; } END { print sum; }')

    mem_util=$(pmap -x $pid | tail -n 1 | awk '{ print $3; }')

    echo "CPU Utilization: $cpu_util%" >> "$result_file"
    echo "Memory Utilization: $mem_util KB" >> "$result_file"

    sleep 1
done