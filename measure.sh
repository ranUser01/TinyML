#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Please provide the Python script file as an argument."
    exit 1
fi

pwd

python_script=$1
current_datetime=$(date +"%Y%m%d%H%M%S")  
result_file="results/$(basename "$python_script" .py)_$current_datetime.txt"

python3 "$python_script" &
pid=$!

while true; do
    if ! kill -0 $pid 2> /dev/null; then
        echo "Process $pid has finished running."
        break
    fi

    cpu_mem_util=$(ps -p $pid -o %cpu,%mem,cmd)
    cpu_util=$(echo $cpu_mem_util | awk '{ print $2 }')
    mem_util=$(echo $cpu_mem_util | awk '{ print $3 }')

    echo "CPU Utilization: $cpu_util%" >> "$result_file"
    echo "Memory Utilization: $mem_util%" >> "$result_file"

    sleep 1
done